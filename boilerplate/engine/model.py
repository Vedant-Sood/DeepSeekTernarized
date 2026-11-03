"""Mode A forward pass for the DeepSeek V4.x family (V4.1 Flash and smaller configs of the same shape).

Layout: the tokens of every sequence in a step are flattened into one ``[1, T, ...]`` batch, so all
projections, the MoE and the hyper-connection math run once over all tokens. Attention is the only
per-sequence part: each sequence attends over its own window ring and its own pages of compressed KV.
A step may mix sequences that are prefilling (``start == 0``, any length) with sequences that are
decoding (one token); chunked prefill (``start > 0`` and more than one token) is not supported yet.

Every numeric step follows the official ``inference/model.py`` (same dtypes, same rounding points),
so on the same kernels the engine reproduces the reference; the comments name the reference piece
each block mirrors. Weights stay in their stored formats: FP8 e4m3 with ue8m0 32x32 block scales,
packed FP4 e2m1 experts with ue8m0 per-32 scales, BF16/FP32 for the small tensors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import torch
import torch.nn.functional as F

from ..checkpoint.dequant import dequant_fp8_block
from ..kernels import torch_ref as K
from .config import EngineConfig
from .engram import EngramHasher, EngramLayout
from .kv_cache import KVCache, Sequence
from .weights import WeightStore

ACT_DTYPE = torch.bfloat16
SCALE_DTYPE = torch.float8_e8m0fnu


# ---------------------------------------------------------------------------------------- linears


class DequantCache:
    """LRU of FP32 dequantized weights for the torch kernel backend, under a byte budget. The
    torch FP8/FP4 GEMMs dequantize the whole weight on every call; caching the result gives the
    same numbers (same tensor, same matmul) without redoing the unpacking for hot weights."""

    def __init__(self, budget_bytes: int):
        from collections import OrderedDict

        self.budget = budget_bytes
        self.used = 0
        # id -> (owner, tensor); holding the owner keeps its id from being reused while cached
        self.items: "OrderedDict[int, tuple[object, torch.Tensor]]" = OrderedDict()

    def get(self, owner, make):
        hit = self.items.get(id(owner))
        if hit is not None:
            self.items.move_to_end(id(owner))
            return hit[1]
        t = make()
        nb = t.numel() * t.element_size()
        if nb <= self.budget:
            while self.used + nb > self.budget and self.items:
                _, (_, old) = self.items.popitem(last=False)
                self.used -= old.numel() * old.element_size()
            self.items[id(owner)] = (owner, t)
            self.used += nb
        return t


class QLinear:
    """y = x @ W^T for a weight in its stored format. FP8/FP4 weights take a block-quantized FP8
    activation (per 32 values, power-of-two scale), as the reference ``linear()`` does; other
    weights run a plain ``F.linear`` in the weight's dtype."""

    cache: DequantCache | None = None  # set by V4Model

    def __init__(self, weight: torch.Tensor, scale: torch.Tensor | None = None, act_block: int = 32):
        self.weight, self.scale, self.act_block = weight, scale, act_block
        if scale is None:
            self.kind = "dense"
        elif weight.dtype == torch.float8_e4m3fn:
            self.kind = "fp8"
            b = act_block
            if (weight.size(0) + b - 1) // b != scale.size(0) or (weight.size(1) + b - 1) // b != scale.size(1):
                # the torch kernel assumes square weight blocks equal to the activation block
                raise ValueError(f"unsupported FP8 block layout {tuple(weight.shape)} / {tuple(scale.shape)}")
        elif weight.dtype in (torch.int8, torch.uint8, torch.float4_e2m1fn_x2):
            self.kind = "fp4"
            self.weight = weight.view(torch.uint8)
        else:
            raise ValueError(f"scaled weight of dtype {weight.dtype}")

    @property
    def out_features(self) -> int:
        return self.weight.size(0)

    def dequantized(self) -> torch.Tensor:
        if self.kind == "fp8":
            make = lambda: K.dequant_fp8_weight(self.weight, self.scale, self.act_block)  # noqa: E731
        else:
            make = lambda: K.dequant_fp4_weight(self.weight, self.scale)  # noqa: E731
        return QLinear.cache.get(self, make) if QLinear.cache is not None else make()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "dense":
            return F.linear(x, self.weight)
        q, s = K.act_quant(x, self.act_block, "ue8m0", SCALE_DTYPE)
        return K.gemm_dequantized(q, s, self.dequantized(), out_dtype=ACT_DTYPE)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    dtype = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
    return (weight * x).to(dtype)


# ---------------------------------------------------------------------------------------- rope


@lru_cache(8)
def rope_table(dim: int, max_pos: int, original_seq_len: int, base: float, factor: float, beta_fast: float, beta_slow: float) -> torch.Tensor:
    """Complex rotations per position; YaRN when original_seq_len > 0 (reference precompute_freqs_cis)."""
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        def corrected_dim(rot):
            return dim * math.log(original_seq_len / (rot * 2 * math.pi)) / (2 * math.log(base))

        low = max(math.floor(corrected_dim(beta_fast)), 0)
        high = min(math.ceil(corrected_dim(beta_slow)), dim - 1)
        ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / max(high - low, 1e-3)).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    freqs = torch.outer(torch.arange(max_pos), freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rope_(x: torch.Tensor, freqs: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """Rotate x in place, adjacent pairs as complex numbers. x: [1, s, d] or [1, s, h, d];
    freqs: [s, d/2]."""
    y = x
    xc = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs = freqs.conj()
    f = freqs.view(1, xc.size(1), xc.size(-1)) if xc.ndim == 3 else freqs.view(1, xc.size(1), 1, xc.size(-1))
    y.copy_(torch.view_as_real(xc * f).flatten(-2))
    return y


# ---------------------------------------------------------------------------------------- step layout


@dataclass
class Segment:
    """One sequence's tokens inside a flattened step."""

    seq: Sequence
    start: int  # absolute position of the first token
    lo: int  # offset in the flattened batch
    n: int  # number of tokens

    @property
    def hi(self) -> int:
        return self.lo + self.n


def window_indices(window: int, seqlen: int, start: int) -> torch.Tensor:
    """Ring slots each query attends to, -1 = empty (reference get_window_topk_idxs), [1, s, k]."""
    if start == 0:
        end = torch.arange(seqlen).unsqueeze(1)
        idxs = (end - window + 1).clamp(0) + torch.arange(min(seqlen, window))
        idxs = torch.where(idxs > end, -1, idxs)
    else:
        oldest = start % window + 1
        idxs = torch.cat([torch.arange(oldest, window), torch.arange(oldest)])
        idxs = torch.where(idxs > start, -1, idxs).unsqueeze(0)
    return idxs.int().unsqueeze(0).contiguous()


def candidate_blocks(logits: torch.Tensor, compress_lens, topk_blocks: int, block_size: int) -> torch.Tensor:
    """Level one of the two-level indexer: per query, the topk_blocks best blocks of positions, the
    block holding the newest position always kept (reference select_candidate_blocks)."""
    width = logits.size(-1)
    scores = F.pad(logits, (0, -width % block_size), value=-torch.inf)
    scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
    nb = scores.size(-1)
    last = (compress_lens - 1) // block_size
    scores = scores.masked_fill(torch.arange(nb) == last, torch.inf)
    top = scores.topk(min(topk_blocks, nb), dim=-1)
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, top.indices, top.values > -torch.inf)
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]


# ---------------------------------------------------------------------------------------- weights


class _NS:
    pass


class V4Model:
    def __init__(self, cfg: EngineConfig, weights: WeightStore, tokenizer=None, max_seq_len: int = 4096, dequant_cache_bytes: int = 2 << 30):
        self.cfg = cfg
        QLinear.cache = DequantCache(dequant_cache_bytes) if dequant_cache_bytes else None
        self.w = weights
        self.max_seq_len = max_seq_len
        blk = cfg.fp8_block
        self.engram_layout = EngramLayout.from_config(cfg.engram)
        self.hasher = None
        if self.engram_layout is not None:
            if tokenizer is None:
                raise ValueError("this checkpoint has engram layers: a tokenizer is needed to hash n-grams")
            self.hasher = EngramHasher(cfg.engram, self.engram_layout, tokenizer)
        W = weights
        g = W.get
        self.embed = g("embed.weight")
        self.norm = g("norm.weight")
        self.head = g("head.weight")
        self.layers = []
        main_rope = rope_table(cfg.rope_head_dim, max_seq_len, 0, cfg.rope_theta, cfg.rope_factor, cfg.beta_fast, cfg.beta_slow)
        comp_rope = rope_table(
            cfg.rope_head_dim, max_seq_len, cfg.original_seq_len, cfg.compress_rope_theta, cfg.rope_factor, cfg.beta_fast, cfg.beta_slow
        )

        def lin(name):
            scale = W.maybe(f"{name}.scale")
            return QLinear(g(f"{name}.weight"), scale, blk)

        for i in range(cfg.n_layers):
            p = f"layers.{i}"
            L = _NS()
            L.id = i
            L.ratio = cfg.compress_ratios[i]
            L.kv_src = cfg.kv_source_of(i)
            L.idx_src = cfg.index_source_of(i)
            L.is_kv_src = i in cfg.kv_source_layers
            L.is_idx_src = i in cfg.index_source_layers and L.ratio > 0
            L.rope = comp_rope if L.ratio else main_rope
            L.attn_norm, L.ffn_norm = g(f"{p}.attn_norm.weight"), g(f"{p}.ffn_norm.weight")
            L.hc = {sub: (g(f"{p}.hc_{sub}_fn").float(), g(f"{p}.hc_{sub}_scale").float(), g(f"{p}.hc_{sub}_base").float()) for sub in ("attn", "ffn")}
            a = f"{p}.attn"
            L.sink = g(f"{a}.attn_sink").float()
            L.wq_a, L.wq_b, L.wkv, L.wo_b = lin(f"{a}.wq_a"), lin(f"{a}.wq_b"), lin(f"{a}.wkv"), lin(f"{a}.wo_b")
            L.q_norm, L.kv_norm = g(f"{a}.q_norm.weight"), g(f"{a}.kv_norm.weight")
            # wo_a is applied per output group (block-diagonal), as an einsum in BF16 over the
            # dequantized weight -- the reference's convert.py dequantizes it the same way
            wo_a, wo_a_s = g(f"{a}.wo_a.weight"), W.maybe(f"{a}.wo_a.scale")
            wo_a = dequant_fp8_block(wo_a, wo_a_s, torch.bfloat16) if wo_a_s is not None else wo_a.bfloat16()
            L.wo_a = wo_a.view(cfg.o_groups, cfg.o_lora_rank, -1)
            if L.is_kv_src:
                c = f"{a}.compressor"
                # ratio 1 is a plain BF16 projection; pooled ratios run in FP32 (reference Compressor)
                cdt = torch.float32 if L.ratio > 1 else torch.bfloat16
                L.c_wkv = QLinear(g(f"{c}.wkv.weight").to(cdt))
                L.c_wgate = QLinear(g(f"{c}.wgate.weight").to(torch.float32)) if L.ratio > 1 else None
                L.c_norm = g(f"{c}.norm.weight")
            if L.is_idx_src:
                x_ = f"{a}.indexer"
                L.i_wq_b = lin(f"{x_}.wq_b")
                L.i_weights = QLinear(g(f"{x_}.weights_proj.weight").bfloat16())
                L.i_owns_k = L.is_kv_src
                if L.i_owns_k:
                    L.i_wk = QLinear(g(f"{x_}.wk.weight").bfloat16())
                    L.i_k_norm = g(f"{x_}.k_norm.weight")
                L.cand_source = i == cfg.candidate_source_layer
                L.uses_cand = 0 <= cfg.candidate_source_layer < i
            f = f"{p}.ffn"
            L.gate_w = g(f"{f}.gate.weight").float()
            L.gate_b = W.maybe(f"{f}.gate.bias")
            L.gate_b = L.gate_b.float() if L.gate_b is not None else None
            L.experts = [None] * cfg.n_routed_experts  # built on first use; weights stay mmap'd
            L.expert_prefix = f"{f}.experts"
            L.shared = tuple(lin(f"{f}.shared_experts.{w}") for w in ("w1", "w2", "w3"))
            L.engram = None
            if self.engram_layout is not None and i in self.engram_layout.layer_ids:
                e = f"{p}.engram"
                E = _NS()
                E.index = self.engram_layout.layer_ids.index(i)
                E.table, E.table_scale = g(f"{e}.embed.weight"), g(f"{e}.embed.scale")
                E.wkv = lin(f"{e}.wkv")
                E.weight = g(f"{e}.q_weight").bfloat16().float() * g(f"{e}.k_weight").bfloat16().float()
                L.engram = E
            self.layers.append(L)
        self._head_f32 = None
        # the reference keeps the head in FP32; converting lazily in vocab chunks keeps memory flat
        self.head_chunk = 32768

    # ---------------------------------------------------------------------------------- pieces

    def expert(self, L, e: int):
        if L.experts[e] is None:
            pre = f"{L.expert_prefix}.{e}"
            L.experts[e] = tuple(
                QLinear(self.w.get(f"{pre}.{w}.weight"), self.w.get(f"{pre}.{w}.scale"), self.cfg.fp8_block) for w in ("w1", "w2", "w3")
            )
        return L.experts[e]

    def swiglu(self, ws, x: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
        w1, w2, w3 = ws
        dtype = x.dtype
        gate, up = w1(x).float(), w3(x).float()
        lim = self.cfg.swiglu_limit
        if lim > 0:
            up = torch.clamp(up, min=-lim, max=lim)
            gate = torch.clamp(gate, max=lim)
        x = F.silu(gate) * up
        if weights is not None:
            x = weights * x
        return w2(x.to(dtype))

    def route(self, L, x: torch.Tensor):
        """x: [n, dim] -> (weights [n, k], expert ids [n, k]) (reference Gate)."""
        cfg = self.cfg
        scores = F.linear(x.float(), L.gate_w) / cfg.gate_temp
        if cfg.score_func == "softmax":
            scores = scores.softmax(dim=-1)
        elif cfg.score_func == "sigmoid":
            scores = scores.sigmoid()
        else:  # sqrtsoftplus
            scores = F.softplus(scores).sqrt()
        biased = scores + L.gate_b if L.gate_b is not None else scores
        indices = biased.topk(cfg.n_activated_experts, dim=-1)[1]
        weights = scores.gather(1, indices)
        if cfg.norm_topk_prob and cfg.n_activated_experts > 1:
            weights /= weights.sum(dim=-1, keepdim=True) + 1e-20
        weights *= cfg.route_scale
        return weights, indices

    def moe(self, L, x: torch.Tensor) -> torch.Tensor:
        shape = x.size()
        x = x.view(-1, self.cfg.dim)
        weights, indices = self.route(L, x)
        y = torch.zeros_like(x, dtype=torch.float32)
        counts = torch.bincount(indices.flatten(), minlength=self.cfg.n_routed_experts).tolist()
        for e, c in enumerate(counts):  # grouped: each expert runs once over all its tokens
            if c == 0:
                continue
            idx, top = torch.where(indices == e)
            y[idx] += self.swiglu(self.expert(L, e), x[idx], weights[idx, top, None])
        y += self.swiglu(L.shared, x)
        return y.type_as(x).view(shape)

    def hc_mixes(self, x: torch.Tensor, fn, scale, base):
        x = x.flatten(2).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.cfg.norm_eps)
        mixes = F.linear(x, fn) * rsqrt
        return K.hc_split_sinkhorn(mixes, scale, base, self.cfg.hc_mult, self.cfg.hc_sinkhorn_iters, self.cfg.hc_eps)

    @staticmethod
    def hc_pre(x: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
        return torch.sum(pre.unsqueeze(-1) * x.float(), dim=2).to(x.dtype)

    @staticmethod
    def hc_post(x, residual, post, comb):
        y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
        return y.type_as(x)

    def engram(self, E, h: torch.Tensor, hash_ids: torch.Tensor) -> torch.Tensor:
        """h: [1, T, hc, dim]; hash_ids: [1, T, n_cols] (reference Engram + ParallelEngramEmbedding)."""
        dim, hc = self.cfg.dim, self.cfg.hc_mult
        flat = hash_ids.reshape(-1)
        rows = E.table.index_select(0, flat).float()
        scales = E.table_scale.index_select(0, flat).float()
        blk = rows.size(-1) // scales.size(-1)
        vals = (rows.unflatten(-1, (-1, blk)) * scales.unsqueeze(-1)).flatten(-2).to(torch.bfloat16)
        vals = vals.view(*hash_ids.shape, -1)
        kv = E.wkv(vals.flatten(-2))
        key, value = kv.split([hc * dim, dim], dim=-1)
        key = key.float().unflatten(-1, (hc, dim))
        x, eps = h.float(), self.cfg.norm_eps
        rstd = torch.rsqrt(x.square().mean(-1) + eps) * torch.rsqrt(key.square().mean(-1) + eps)
        dot = (x * E.weight * key).sum(-1) * rstd * dim**-0.5
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
        return (x + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(h.dtype)

    # ---------------------------------------------------------------------------------- attention

    def attention(self, L, x: torch.Tensor, segs: list[Segment], positions: torch.Tensor, cache: KVCache, shared: dict) -> torch.Tensor:
        cfg = self.cfg
        rd, H, hd = cfg.rope_head_dim, cfg.n_heads, cfg.head_dim
        freqs = L.rope[positions]  # [T, rd/2]
        qr = rms_norm(L.wq_a(x), L.q_norm, cfg.norm_eps)
        q = L.wq_b(qr).unflatten(-1, (H, hd))
        apply_rope_(q[..., -rd:], freqs)
        kv = rms_norm(L.wkv(x), L.kv_norm, cfg.norm_eps)
        apply_rope_(kv[..., -rd:], freqs)
        K.act_quant(kv, cfg.fp8_block, "ue8m0", SCALE_DTYPE, True)

        # projections that run over every token of the step at once
        comp = None
        if L.is_kv_src:
            if L.ratio > 1:
                xf = x.float()
                comp = (L.c_wkv(xf), L.c_wgate(xf))
            else:
                comp = (rms_norm(L.c_wkv(x), L.c_norm, cfg.norm_eps), None)
        idx_q = idx_w = None
        if L.is_idx_src:
            idx_q = L.i_wq_b(qr).unflatten(-1, (cfg.index_n_heads, cfg.index_head_dim))
            apply_rope_(idx_q[..., -rd:], freqs)
            K.fp4_act_quant(idx_q, 32, True)
            idx_w = L.i_weights(x) * (cfg.index_head_dim**-0.5 * cfg.index_n_heads**-0.5)

        o = torch.empty_like(q)
        for sg in segs:
            sl = slice(sg.lo, sg.hi)
            o[:, sl] = self._attend_one(L, sg, q[:, sl], kv[:, sl], comp, idx_q, idx_w, cache, shared)

        apply_rope_(o[..., -rd:], freqs, inverse=True)
        T = x.size(1)
        o = o.view(1, T, cfg.o_groups, -1)
        K.act_quant(o, cfg.fp8_block, "ue8m0", SCALE_DTYPE, True)
        o = torch.einsum("bsgd,grd->bsgr", o, L.wo_a)
        return L.wo_b(o.flatten(2))

    def _attend_one(self, L, sg: Segment, q, kv, comp, idx_q, idx_w, cache: KVCache, shared: dict):
        cfg, seq, start, n = self.cfg, sg.seq, sg.start, sg.n
        W, rd = cfg.window_size, cfg.rope_head_dim
        ring = cache.window[L.id][seq.slot]
        if start == 0:
            if n <= W:
                ring[:n] = kv[0]
            else:
                cut = n % W
                ring[cut:W], ring[:cut] = kv[0, -W:].split([W - cut, cut])
            window_kv = kv
        else:
            if n != 1:
                raise NotImplementedError("chunked prefill (start > 0 with several tokens) is not supported yet")
            ring[start % W] = kv[0, 0]
            window_kv = ring.unsqueeze(0)
        idxs = window_indices(W, n, start)
        if L.ratio:
            r = L.ratio
            end = start + n
            clen = end // r
            latent = None
            if L.is_kv_src:
                latent = self._compress(L, sg, comp, cache)
            # top-k compressed positions: index sources compute them, the layers after reuse them
            if L.is_idx_src:
                if clen == 0:
                    cidx = torch.empty(1, n, 0, dtype=torch.int32)
                else:
                    cidx = self._index(L, sg, latent, idx_q, idx_w, window_kv.size(1), cache, shared)
                shared[("topk", seq.seq_id)] = cidx
            else:
                cidx = shared[("topk", seq.seq_id)]
            if latent is not None:
                fr = L.rope[: n - n % r : r] if start == 0 else L.rope[start + 1 - r].unsqueeze(0)
                apply_rope_(latent[..., -rd:], fr)
                K.fp4_act_quant(latent, 16, True, scale_dtype=torch.float8_e4m3fn)
                cache.write_compressed(cache.ckv[L.kv_src], seq, r, start // r, latent[0])
            ckv = cache.read_compressed(cache.ckv[L.kv_src], seq, r, clen).unsqueeze(0)
            window_kv = torch.cat([window_kv, ckv], dim=1)
            idxs = torch.cat([idxs, cidx], dim=-1)
        return K.sparse_attn(q, window_kv, L.sink, idxs, cfg.head_dim**-0.5)

    def _compress(self, L, sg: Segment, comp, cache: KVCache):
        """Pool ratio consecutive tokens into one latent (pre-RoPE), or None while a group is
        incomplete (reference Compressor). Returns [1, groups, head_dim] BF16."""
        cfg, r, seq, start, n = self.cfg, L.ratio, sg.seq, sg.start, sg.n
        kv_all, score_all = comp
        kv = kv_all[:, sg.lo : sg.hi]
        if r == 1:
            return kv.clone()
        score = score_all[:, sg.lo : sg.hi]
        tail_kv, tail_s = cache.tail_kv[L.id][seq.slot], cache.tail_score[L.id][seq.slot]
        if start == 0:
            rem = n % r
            cut = n - rem
            if rem:
                tail_kv[:rem], tail_s[:rem] = kv[0, cut:], score[0, cut:]
            if n < r:
                return None
            kv = kv[:, :cut].unflatten(1, (-1, r))
            score = score[:, :cut].unflatten(1, (-1, r))
            pooled = (kv * score.softmax(dim=2)).sum(dim=2)
        else:
            slot = start % r
            tail_kv[slot], tail_s[slot] = kv[0, 0], score[0, 0]
            if (start + 1) % r:
                return None
            pooled = (tail_kv.unsqueeze(0) * tail_s.unsqueeze(0).softmax(dim=1)).sum(dim=1, keepdim=True)
        return rms_norm(pooled.to(ACT_DTYPE), L.c_norm, cfg.norm_eps)

    def _index(self, L, sg: Segment, latent, idx_q, idx_w, offset: int, cache: KVCache, shared: dict):
        """Top-k compressed positions per query (reference Indexer + select_candidate_blocks)."""
        cfg, r, seq, start, n = self.cfg, L.ratio, sg.seq, sg.start, sg.n
        rd, end = cfg.rope_head_dim, sg.start + sg.n
        pool = cache.ik[L.kv_src]
        if L.i_owns_k and latent is not None:
            fr = L.rope[: n - n % r : r] if start == 0 else L.rope[start + 1 - r].unsqueeze(0)
            k = rms_norm(L.i_wk(latent), L.i_k_norm, cfg.norm_eps)
            apply_rope_(k[..., -rd:], fr)
            K.fp4_act_quant(k, 32, True)
            cache.write_compressed(pool, seq, r, start // r, k[0])
        q = idx_q[:, sg.lo : sg.hi]
        w = idx_w[:, sg.lo : sg.hi]
        index_k = cache.read_compressed(pool, seq, r, end // r).unsqueeze(0)
        score = torch.einsum("bshd,btd->bsht", q, index_k)
        score = (score.relu_() * w.unsqueeze(-1)).sum(dim=2)
        if start == 0:
            clens = (torch.arange(1, n + 1) // r).unsqueeze(-1)
            score.masked_fill_(torch.arange(n // r) >= clens, -torch.inf)
        else:
            clens = end // r
        if L.cand_source:
            shared[("cand", seq.seq_id)] = candidate_blocks(score, clens, cfg.candidate_topk_blocks, cfg.candidate_block_size)
        elif L.uses_cand:
            score = score.masked_fill(~shared[("cand", seq.seq_id)], -torch.inf)
        topk = min(cfg.index_topk, end // r)
        idxs = score.topk(topk, dim=-1, sorted=False).indices.sort(dim=-1).values
        return torch.where(idxs < clens, idxs + offset, -1).int()

    # ---------------------------------------------------------------------------------- forward

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        """h: [..., dim] BF16 after the final norm -> FP32 logits (the reference keeps the head in FP32)."""
        x = h.float()
        if self.head.size(0) <= self.head_chunk:
            if self._head_f32 is None:
                self._head_f32 = self.head.float()
            return F.linear(x, self._head_f32)
        return torch.cat([F.linear(x, self.head[i : i + self.head_chunk].float()) for i in range(0, self.head.size(0), self.head_chunk)], -1)

    @torch.inference_mode()
    def forward(
        self,
        chunks: list[tuple[Sequence, list[int]]],
        cache: KVCache,
        all_logits: bool = False,
        capture: list | None = None,
    ) -> torch.Tensor:
        """Run one step. chunks: (sequence, new token ids) pairs. Returns FP32 logits for the last
        token of each chunk ([n_chunks, vocab]) or for every token ([T, vocab]). When ``capture`` is a
        list, the residual stream after every layer ([T, hc, dim]) is appended to it."""
        cfg = self.cfg
        segs, ids, pos, hashes = [], [], [], []
        lo = 0
        for seq, toks in chunks:
            n = len(toks)
            if seq.length + n > self.max_seq_len:
                raise ValueError(f"sequence would reach {seq.length + n} tokens, max_seq_len is {self.max_seq_len}")
            cache.reserve(seq, seq.length + n)
            segs.append(Segment(seq, seq.length, lo, n))
            t = torch.tensor(toks, dtype=torch.long)
            ids.append(t)
            pos.append(torch.arange(seq.length, seq.length + n))
            if self.hasher is not None:
                hid, seq.engram_history = self.hasher(t, seq.engram_history)
                hashes.append(hid)
            lo += n
        ids_t, positions = torch.cat(ids), torch.cat(pos)
        hash_ids = torch.cat(hashes).unsqueeze(0) if hashes else None  # [1, T, L_engram, cols]

        h = F.embedding(ids_t, self.embed).unsqueeze(0)  # [1, T, dim]
        h = h.unsqueeze(2).repeat(1, 1, cfg.hc_mult, 1)
        pre = h.new_zeros(1, h.size(1), cfg.hc_mult, dtype=torch.float32)
        pre[..., 0] = 1.0
        shared: dict = {}
        for L in self.layers:
            if L.engram is not None:
                h = self.engram(L.engram, h, hash_ids[:, :, L.engram.index])
            residual = h
            a_pre, a_post, a_comb = self.hc_mixes(h, *L.hc["attn"])
            x = rms_norm(self.hc_pre(h, pre), L.attn_norm, cfg.norm_eps)
            x = self.attention(L, x, segs, positions, cache, shared)
            h = self.hc_post(x, residual, a_post, a_comb)
            residual = h
            f_pre, f_post, f_comb = self.hc_mixes(h, *L.hc["ffn"])
            x = rms_norm(self.hc_pre(h, a_pre), L.ffn_norm, cfg.norm_eps)
            x = self.moe(L, x)
            h = self.hc_post(x, residual, f_post, f_comb)
            pre = f_pre
            if capture is not None:
                capture.append(h[0].clone())
        for sg in segs:
            sg.seq.length += sg.n
        h = rms_norm(self.hc_pre(h, pre), self.norm, cfg.norm_eps)[0]  # [T, dim]
        if not all_logits:
            h = h[torch.tensor([sg.hi - 1 for sg in segs])]
        return self.logits(h)

"""CPU correctness path for config-driven Llama and OLMoE causal decoders.

The checkpoint's config and tensor names determine dimensions and routing. This first path is
deliberately simple: attention uses PyTorch and the KV cache is a per-layer tensor pair. It shares
the existing weight store, linear and ternary formats with the V4 engine.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from ..reference.trace import ReferenceTrace
from ..ternary.ops import PackedTerm, TernaryLinear
from .model import QLinear
from .weights import TensorRef, WeightStore


def _value(weight):
    return weight.get() if isinstance(weight, TensorRef) else weight


class _Linear:
    def __init__(self, store: WeightStore, base: str, dtype: torch.dtype):
        self.meta = store.mode_of(base)
        self.bias = store.maybe_ref(base + ".bias")
        self.ternary = None
        if self.meta["mode"] in ("C", "D"):
            terms = [PackedTerm(store.get(f"{base}.tq{i}.packed"), store.get(f"{base}.tq{i}.scale"))
                     for i in range(self.meta["terms"])]
            self.ternary = TernaryLinear(terms, self.meta["shape"][1], self.meta["packing"],
                                         self.meta["group"], self.meta["act"])
            self.linear = None
        else:
            self.linear = QLinear(store.ref(base + ".weight"), weight_dtype=dtype)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        y = self.ternary(x, out_dtype=x.dtype) if self.ternary is not None else self.linear(x)
        return y if self.bias is None else y + _value(self.bias).to(y.dtype)


class StandardDecoder:
    def __init__(self, path: str | Path, *, dtype: torch.dtype = torch.float32,
                 mode_b_cache_bytes: int = 256 << 20):
        self.path = Path(path)
        self.config = json.loads((self.path / "config.json").read_text())
        c = self.config
        if c.get("model_type") not in ("llama", "olmoe"):
            raise ValueError(f"unsupported standard decoder model_type {c.get('model_type')!r}")
        if c.get("hidden_act", "silu") != "silu" or c.get("rope_scaling") or c.get("rope_interleaved", False):
            raise ValueError("only SiLU and non-interleaved, unscaled RoPE are supported by this correctness path")
        self.dtype = dtype
        self.hidden = c["hidden_size"]
        self.heads = c["num_attention_heads"]
        self.kv_heads = c.get("num_key_value_heads", self.heads)
        self.head_dim = c.get("head_dim", self.hidden // self.heads)
        if self.heads % self.kv_heads or self.head_dim % 2:
            raise ValueError("invalid attention head configuration")
        self.eps = c.get("rms_norm_eps", 1e-6)
        self.theta = c.get("rope_theta", 10000.0)
        self.store = WeightStore(path, mode_b_cache_bytes=mode_b_cache_bytes)
        self.embedding = self.store.ref("model.embed_tokens.weight")
        head_name = "lm_head" if "lm_head.weight" in self.store or self.store.mode_of("lm_head")["mode"] in ("C", "D") else None
        if head_name is None and not c.get("tie_word_embeddings", False):
            raise ValueError("checkpoint lacks an LM head and does not declare tied embeddings")
        self.head = _Linear(self.store, head_name, dtype) if head_name else None
        self.final_norm = self.store.ref("model.norm.weight")
        self.layers = []
        for index in range(c["num_hidden_layers"]):
            base = f"model.layers.{index}"
            attn = base + ".self_attn"
            layer = SimpleNamespace(
                input_norm=self.store.ref(base + ".input_layernorm.weight"),
                post_norm=self.store.ref(base + ".post_attention_layernorm.weight"),
                q=_Linear(self.store, attn + ".q_proj", dtype),
                k=_Linear(self.store, attn + ".k_proj", dtype),
                v=_Linear(self.store, attn + ".v_proj", dtype),
                o=_Linear(self.store, attn + ".o_proj", dtype),
                q_norm=self.store.maybe_ref(attn + ".q_norm.weight"),
                k_norm=self.store.maybe_ref(attn + ".k_norm.weight"),
                prefix=base + ".mlp", experts={},
            )
            if (layer.q_norm is None) != (layer.k_norm is None):
                raise ValueError("both Q and K normalization weights must be present")
            if c["model_type"] == "llama":
                layer.mlp = self._mlp(layer.prefix)
            else:
                layer.router = _Linear(self.store, layer.prefix + ".gate", dtype)
            self.layers.append(layer)

    def _mlp(self, base):
        return (_Linear(self.store, base + ".gate_proj", self.dtype),
                _Linear(self.store, base + ".up_proj", self.dtype),
                _Linear(self.store, base + ".down_proj", self.dtype))

    def _norm(self, x, weight):
        xf = x.float()
        y = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.eps)
        return y.to(x.dtype) * _value(weight).to(x.dtype)

    def _rope(self, q, k, offset):
        positions = torch.arange(offset, offset + q.size(-2), dtype=torch.float32, device=q.device)
        inverse = self.theta ** (-torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=q.device) / self.head_dim)
        frequencies = torch.outer(positions, inverse)
        angles = torch.cat((frequencies, frequencies), dim=-1)
        cos, sin = angles.cos().to(q.dtype)[None, None], angles.sin().to(q.dtype)[None, None]
        def rotate(x):
            a, b = x.chunk(2, dim=-1)
            return torch.cat((-b, a), dim=-1)
        return q * cos + rotate(q) * sin, k * cos + rotate(k) * sin

    @staticmethod
    def _run_mlp(mlp, x):
        gate, up, down = mlp
        return down(F.silu(gate(x)) * up(x))

    def _moe(self, layer, x):
        shape = x.shape
        flat = x.reshape(-1, self.hidden)
        scores = torch.softmax(layer.router(flat).float(), dim=-1)
        values, indices = torch.topk(scores, self.config["num_experts_per_tok"], dim=-1)
        if self.config.get("norm_topk_prob", False):
            values = values / values.sum(-1, keepdim=True)
        values = values.to(x.dtype)
        output = torch.zeros_like(flat)
        for expert in range(self.config["num_experts"]):
            token, slot = torch.where(indices == expert)
            if not token.numel():
                continue
            if expert not in layer.experts:
                layer.experts[expert] = self._mlp(f"{layer.prefix}.experts.{expert}")
            result = self._run_mlp(layer.experts[expert], flat[token]) * values[token, slot, None]
            output.index_add_(0, token, result.to(output.dtype))
        return output.view(shape)

    @torch.inference_mode()
    def forward(self, ids: torch.Tensor, past=None, *, capture_layers: list | None = None):
        if ids.ndim != 2 or ids.device.type != "cpu" or ids.shape[1] == 0:
            raise ValueError("input IDs must be a nonempty CPU [batch, sequence] tensor")
        past = [None] * len(self.layers) if past is None else past
        if len(past) != len(self.layers):
            raise ValueError("KV cache layer count differs from the model")
        x = (self.store.rows(self.embedding.name, ids) if isinstance(self.embedding, TensorRef)
             else F.embedding(ids, self.embedding)).to(self.dtype)
        states, new_past = [x], []
        for index, layer in enumerate(self.layers):
            residual = x
            z = self._norm(x, layer.input_norm)
            q, k, v = layer.q(z), layer.k(z), layer.v(z)
            if layer.q_norm is not None:
                q, k = self._norm(q, layer.q_norm), self._norm(k, layer.k_norm)
            if self.config.get("clip_qkv") is not None:
                bound = self.config["clip_qkv"]
                q, k, v = q.clamp(-bound, bound), k.clamp(-bound, bound), v.clamp(-bound, bound)
            b, s, _ = q.shape
            q = q.view(b, s, self.heads, self.head_dim).transpose(1, 2)
            k = k.view(b, s, self.kv_heads, self.head_dim).transpose(1, 2)
            v = v.view(b, s, self.kv_heads, self.head_dim).transpose(1, 2)
            offset = 0 if past[index] is None else past[index][0].size(-2)
            q, k = self._rope(q, k, offset)
            if past[index] is not None:
                k = torch.cat((past[index][0], k), dim=-2)
                v = torch.cat((past[index][1], v), dim=-2)
            new_past.append((k, v))
            repeat = self.heads // self.kv_heads
            keys, vals = k.repeat_interleave(repeat, dim=1), v.repeat_interleave(repeat, dim=1)
            scores = (q @ keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
            mask = torch.arange(k.size(-2), device=ids.device)[None, :] > torch.arange(offset, offset + s, device=ids.device)[:, None]
            scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
            attention = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype) @ vals
            x = residual + layer.o(attention.transpose(1, 2).reshape(b, s, -1).contiguous())
            residual = x
            z = self._norm(x, layer.post_norm)
            x = residual + (self._run_mlp(layer.mlp, z) if self.config["model_type"] == "llama" else self._moe(layer, z))
            if capture_layers is not None:
                capture_layers.append(x)
            states.append(x)
        x = self._norm(x, self.final_norm)
        states[-1] = x
        logits = self.head(x) if self.head is not None else F.linear(x, _value(self.embedding).to(x.dtype))
        return logits.float(), states, new_past


class StandardEngine:
    def __init__(self, model: StandardDecoder, tokenizer=None, *, max_seq_len=4096, max_seqs=8, max_tokens=16384):
        if min(max_seq_len, max_seqs, max_tokens) <= 0:
            raise ValueError("cache limits must be positive")
        self.model, self.tokenizer = model, tokenizer
        self.cfg = SimpleNamespace(eos_token_id=model.config.get("eos_token_id"), vocab_size=model.config["vocab_size"])
        self.max_seq_len = min(max_seq_len, model.config.get("max_position_embeddings", max_seq_len))
        self.max_seqs, self.max_tokens = max_seqs, max_tokens

    @classmethod
    def load(cls, path: str | Path, *, dtype: torch.dtype = torch.float32, max_seq_len=4096,
             mode_b_cache_bytes: int = 256 << 20, tokenizer_path=None, max_seqs=8, max_tokens=16384):
        from .engine import load_tokenizer
        return cls(StandardDecoder(path, dtype=dtype, mode_b_cache_bytes=mode_b_cache_bytes),
                   load_tokenizer(tokenizer_path or path), max_seq_len=max_seq_len,
                   max_seqs=max_seqs, max_tokens=max_tokens)

    def trace(self, input_ids: torch.Tensor, decode_from: int | None = None) -> ReferenceTrace:
        if (input_ids.ndim != 2 or input_ids.shape[0] > self.max_seqs or input_ids.shape[1] > self.max_seq_len
                or input_ids.numel() > self.max_tokens):
            raise ValueError("input exceeds standard decoder cache limits")
        k = input_ids.shape[1] if decode_from is None else decode_from
        if not 0 < k <= input_ids.shape[1]:
            raise ValueError("decode_from must be within the sequence")
        logits, states, past = self.model.forward(input_ids[:, :k])
        all_logits, all_states = [logits], [[x] for x in states]
        for t in range(k, input_ids.shape[1]):
            logits, states, past = self.model.forward(input_ids[:, t:t + 1], past)
            all_logits.append(logits)
            for acc, state in zip(all_states, states):
                acc.append(state)
        return ReferenceTrace(input_ids, [torch.cat(s, dim=1) for s in all_states], torch.cat(all_logits, dim=1),
                              {"backend": "boilerplate-standard", "decode_from": decode_from,
                               "hidden_semantics": "transformers hidden_states"})

    def generate(self, prompts, max_new_tokens=32, temperature=0.0, seed=None, eos_token_ids=None, stream=None):
        from .engine import GenerationResult, sample
        if (max_new_tokens < 0 or len(prompts) > self.max_seqs or
                any(not p or len(p) + max_new_tokens > self.max_seq_len for p in prompts) or
                sum(len(p) + max_new_tokens for p in prompts) > self.max_tokens):
            raise ValueError("generation exceeds standard decoder cache limits or has an empty prompt")
        eos = self.cfg.eos_token_id
        eos_token_ids = eos_token_ids if eos_token_ids is not None else (set(eos) if isinstance(eos, list) else {eos} if eos is not None else set())
        generator = torch.Generator().manual_seed(seed) if seed is not None else None
        results, finished = [], []
        for index, prompt in enumerate(prompts):
            tokens, done, past = [], False, None
            if max_new_tokens:
                logits, _, past = self.model.forward(torch.tensor([prompt], dtype=torch.long))
                for step in range(max_new_tokens):
                    token = int(sample(logits[:, -1], temperature, generator).item())
                    tokens.append(token)
                    if stream is not None:
                        stream(index, token)
                    if token in eos_token_ids:
                        done = True
                        break
                    if step + 1 < max_new_tokens:
                        logits, _, past = self.model.forward(torch.tensor([[token]]), past)
            results.append(tokens)
            finished.append(done)
        return GenerationResult(results, finished)

    def generate_text(self, prompt: str, **kwargs) -> str:
        if self.tokenizer is None:
            raise ValueError("no tokenizer.json next to the checkpoint")
        return self.tokenizer.decode(self.generate([self.tokenizer.encode(prompt).ids], **kwargs).tokens[0])

"""Pure-PyTorch kernels with the numerics of the official ``inference/kernel.py`` (TileLang, GPU-only).

Two users: the reference harness swaps them in for the TileLang module so the unmodified official
code runs on CPU (``boilerplate.reference.cpu_kernels`` re-exports them under the reference's names),
and the Boilerplate engine uses them as its Mode A "torch" kernel backend.

Same function names, signatures and numerics contract, written for clarity on CPU (or any device):

* ``act_quant``       -- block FP8 e4m3 activation quantization; power-of-two scales when scale_fmt set
* ``fp4_act_quant``   -- block FP4 e2m1 quantization with ue8m0 or e4m3 scales
* ``fp8_gemm``        -- C = A_fp8 @ B_fp8^T with per-row/K-block and per-(N,K)-block scales
* ``fp4_gemm``        -- C = A_fp8 @ B_fp4^T with per-32 weight scales
* ``sparse_attn``     -- gathered-index attention with a learned per-head sink logit
* ``hc_split_sinkhorn`` -- hyper-connection pre/post/comb mixes, comb made doubly stochastic

Differences from the GPU kernels are confined to floating-point summation order (the kernels use
blocked online softmax and per-K-block accumulation). The probability tile is rounded to BF16 before
the PV product, as in the kernel. These functions are the semantic reference our own engine kernels
are tested against; they are not meant to be fast.
"""

from __future__ import annotations

import torch

from ..checkpoint.dequant import FP4_MAX, FP8_E4M3_MAX, pack_fp4, pow2_ceil, round_to_fp4, unpack_fp4


def _blocks(x: torch.Tensor, block: int) -> torch.Tensor:
    return x.float().unflatten(-1, (-1, block))


def act_quant(
    x: torch.Tensor,
    block_size: int = 128,
    scale_fmt: str | None = None,
    scale_dtype: torch.dtype = torch.float32,
    inplace: bool = False,
):
    N = x.size(-1)
    assert N % block_size == 0
    xb = _blocks(x, block_size)
    amax = xb.abs().amax(-1).clamp(min=1e-4)
    s = pow2_ceil(amax / FP8_E4M3_MAX) if scale_fmt is not None else amax / FP8_E4M3_MAX
    q = (xb / s.unsqueeze(-1)).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    if inplace:
        y = (q.float() * s.unsqueeze(-1)).flatten(-2).to(x.dtype)
        x.copy_(y)
        return x
    return q.flatten(-2), s.to(scale_dtype)


def fp4_act_quant(
    x: torch.Tensor,
    block_size: int = 32,
    inplace: bool = False,
    scale_dtype: torch.dtype = torch.float8_e8m0fnu,
):
    assert scale_dtype in (torch.float8_e8m0fnu, torch.float8_e4m3fn)
    N = x.size(-1)
    assert N % block_size == 0
    xb = _blocks(x, block_size)
    amax = xb.abs().amax(-1)
    if scale_dtype == torch.float8_e4m3fn:
        amax = amax.clamp(min=FP4_MAX * 2.0**-9)
        s = (amax / FP4_MAX).to(torch.float8_e4m3fn).float()
    else:
        amax = amax.clamp(min=FP4_MAX * 2.0**-126)
        s = pow2_ceil(amax / FP4_MAX)
    codes = round_to_fp4((xb / s.unsqueeze(-1)).clamp(-FP4_MAX, FP4_MAX))
    if inplace:
        from ..checkpoint.dequant import FP4_VALUES

        y = (FP4_VALUES[codes] * s.unsqueeze(-1)).flatten(-2).to(x.dtype)
        x.copy_(y)
        return x
    packed = pack_fp4(codes.flatten(-2)).view(torch.float4_e2m1fn_x2)
    return packed, s.to(scale_dtype)


def _expand_act_scale(a_s: torch.Tensor, K: int) -> torch.Tensor:
    a_s = a_s.float().reshape(-1, a_s.size(-1))
    return a_s.repeat_interleave(K // a_s.size(-1), dim=-1)


def dequant_fp8_weight(b: torch.Tensor, b_s: torch.Tensor, block_size: int) -> torch.Tensor:
    """FP32 copy of an FP8 weight with (block x block) scales: what fp8_gemm multiplies by."""
    N, K = b.shape
    sb = b_s.float().repeat_interleave(block_size, 0)[:N].repeat_interleave(block_size, 1)[:, :K]
    return b.float() * sb


def dequant_fp4_weight(b: torch.Tensor, b_s: torch.Tensor) -> torch.Tensor:
    """FP32 copy of a packed FP4 weight with per-32 scales: what fp4_gemm multiplies by."""
    N = b.size(0)
    assert b_s.shape == (N, b.size(1) * 2 // 32)
    return unpack_fp4(b) * b_s.float().repeat_interleave(32, dim=1)


def gemm_dequantized(a, a_s, B: torch.Tensor, out_dtype=None):
    """C = dequant(A) @ B^T with A FP8 + per-block scales and B an already dequantized FP32 weight.
    fp8_gemm / fp4_gemm are this with B dequantized on the fly; callers may cache B."""
    K = a.size(-1)
    M = a.numel() // K
    A = a.reshape(M, K).float() * _expand_act_scale(a_s, K)
    c = A @ B.t()
    return c.reshape(*a.shape[:-1], B.size(0)).to(out_dtype or torch.get_default_dtype())


def fp8_gemm(a, a_s, b, b_s, scale_dtype: torch.dtype = torch.float32, block_size: int = 128, out_dtype=None):
    assert a.size(-1) % block_size == 0
    return gemm_dequantized(a, a_s, dequant_fp8_weight(b, b_s, block_size), out_dtype)


def fp4_gemm(a, a_s, b, b_s, scale_dtype: torch.dtype = torch.float32, act_block_size: int = 128, out_dtype=None):
    return gemm_dequantized(a, a_s, dequant_fp4_weight(b, b_s), out_dtype)


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale: float):
    """q: [b, m, h, d] bf16; kv: [b, n, d] bf16 (shared by all heads); topk_idxs: [b, m, k] int32,
    -1 = empty slot. Softmax over the selected positions plus a per-head sink logit that takes
    probability mass but contributes no value. A row with no valid index returns zeros."""
    b, m, h, d = q.shape
    idx = topk_idxs.long()
    valid = idx >= 0
    gathered = torch.gather(
        kv.unsqueeze(1).expand(b, m, kv.size(1), d), 2, idx.clamp(min=0).unsqueeze(-1).expand(b, m, idx.size(-1), d)
    )  # [b, m, k, d]
    gathered = torch.where(valid.unsqueeze(-1), gathered, torch.zeros_like(gathered))
    scores = torch.einsum("bmhd,bmkd->bmhk", q.float(), gathered.float()) * softmax_scale
    scores = scores.masked_fill(~valid.unsqueeze(2), float("-inf"))
    smax = scores.amax(-1).clamp(min=-1e30)  # finite floor, as in the kernel
    p = torch.exp(scores - smax.unsqueeze(-1))
    denom = p.sum(-1) + torch.exp(attn_sink.float().view(1, 1, h) - smax)
    o = torch.einsum("bmhk,bmkd->bmhd", p.to(torch.bfloat16).float(), gathered.float())
    return (o / denom.unsqueeze(-1)).to(q.dtype)


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult: int = 4, sinkhorn_iters: int = 20, eps: float = 1e-6):
    hc = hc_mult
    m = mixes.float()
    scale, base = hc_scale.float(), hc_base.float()
    pre = torch.sigmoid(m[..., :hc] * scale[0] + base[:hc]) + eps
    post = 2 * torch.sigmoid(m[..., hc : 2 * hc] * scale[1] + base[hc : 2 * hc])
    comb = (m[..., 2 * hc :] * scale[2] + base[2 * hc :]).unflatten(-1, (hc, hc))
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb

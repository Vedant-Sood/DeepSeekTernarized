"""Block-scaled FP8 / FP4 storage formats used by DeepSeek V4.x checkpoints.

* FP8 weights: ``float8_e4m3fn`` with one scale per ``(bo, bi)`` block (32x32 in V4.1), scales in
  ``float8_e8m0fnu`` (power-of-two, "ue8m0").
* FP4 weights: e2m1 values packed two per byte along the input dim (low nibble = even index),
  stored as int8 ``[out, in // 2]`` (or ``float4_e2m1fn_x2``), one ue8m0 scale per 32 inputs.

Block sizes are inferred from the weight/scale shapes, never assumed.
"""

from __future__ import annotations

import torch

FP4_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=torch.float32
)
FP8_E4M3_MAX = 448.0
FP4_MAX = 6.0


def infer_block(weight_shape: tuple[int, ...], scale_shape: tuple[int, ...]) -> tuple[int, int]:
    """Block size (bo, bi) such that scale = ceil(weight / block) on each axis.

    When a dim is not a multiple of the block, several sizes fit (70 rows in 3 blocks: 24..34);
    prefer a power of two in the feasible range, as real block sizes are."""

    def axis(n: int, s: int) -> int:
        lo = -(-n // s)
        hi = n if s == 1 else (n - 1) // (s - 1)
        if lo > hi or -(-n // lo) != s:
            raise ValueError(f"cannot infer block size from weight {weight_shape} and scale {scale_shape}")
        p = 1 << (hi.bit_length() - 1)  # largest power of two <= hi
        return p if p >= lo else lo

    (o, i), (so, si) = weight_shape, scale_shape
    return axis(o, so), axis(i, si)


def dequant_fp8_block(weight: torch.Tensor, scale: torch.Tensor, out_dtype=torch.bfloat16) -> torch.Tensor:
    o, i = weight.shape
    bo, bi = infer_block(tuple(weight.shape), tuple(scale.shape))
    s = scale.float().repeat_interleave(bo, 0)[:o].repeat_interleave(bi, 1)[:, :i]
    return (weight.float() * s).to(out_dtype)


def unpack_fp4(packed: torch.Tensor) -> torch.Tensor:
    """[out, in//2] int8/uint8/float4_e2m1fn_x2 -> [out, in] float32 e2m1 values (unscaled)."""
    b = packed.view(torch.uint8)
    lo, hi = (b & 0x0F).long(), (b >> 4).long()
    return torch.stack([FP4_VALUES[lo], FP4_VALUES[hi]], dim=-1).flatten(-2)


def is_packed_fp4(weight_shape: tuple[int, ...], scale_shape: tuple[int, ...], group: int = 32) -> bool:
    """An I8 weight is packed FP4 when its scale covers 2x its stored columns in groups of ``group``."""
    return (
        len(weight_shape) == 2
        and len(scale_shape) == 2
        and scale_shape[0] == weight_shape[0]
        and scale_shape[1] * group == weight_shape[1] * 2
    )


def dequant_fp4(packed: torch.Tensor, scale: torch.Tensor, out_dtype=torch.bfloat16) -> torch.Tensor:
    vals = unpack_fp4(packed)
    group = vals.shape[1] // scale.shape[1]
    return (vals * scale.float().repeat_interleave(group, 1)).to(out_dtype)


def round_to_fp4(x: torch.Tensor) -> torch.Tensor:
    """Round float32 values (already divided by scale, |x| <= 6) to the nearest e2m1 value,
    ties to even mantissa -- the IEEE behaviour of a hardware cvt. Returns codes 0..15."""
    a = x.abs().clamp(max=FP4_MAX)
    # magnitudes 0, .5, 1, 1.5, 2, 3, 4, 6 -> codes 0..7; midpoints, and which side wins a tie
    mids = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
    tie_up = torch.tensor([False, True, False, True, False, True, False])  # tie goes to even code
    code = torch.zeros_like(a, dtype=torch.long)
    for k, (m, up) in enumerate(zip(mids.tolist(), tie_up.tolist())):
        code = torch.where((a > m) | ((a == m) & up), torch.full_like(code, k + 1), code)
    sign = (x < 0) & (code > 0)
    return code + sign.long() * 8


def pack_fp4(codes: torch.Tensor) -> torch.Tensor:
    """[out, in] codes 0..15 -> [out, in//2] int8, low nibble = even index."""
    c = codes.to(torch.uint8)
    return (c[..., 0::2] | (c[..., 1::2] << 4)).view(torch.int8)


def pow2_ceil(x: torch.Tensor) -> torch.Tensor:
    """2**ceil(log2(x)) for positive float32, computed exactly like the reference kernels
    (exponent bits + 1 if any mantissa bit is set)."""
    bits = x.float().contiguous().view(torch.int32)
    exp = ((bits >> 23) & 0xFF) - 127
    exp = exp + ((bits & 0x7FFFFF) != 0).int()
    return torch.ldexp(torch.ones_like(x, dtype=torch.float32), exp)


def quant_fp8_block(w: torch.Tensor, block: tuple[int, int] = (32, 32)) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D weight to e4m3 with power-of-two (ue8m0) per-block scales."""
    o, i = w.shape
    bo, bi = block
    so, si = -(-o // bo), -(-i // bi)
    wp = torch.zeros(so * bo, si * bi, dtype=torch.float32)
    wp[:o, :i] = w.float()
    blocks = wp.view(so, bo, si, bi)
    amax = blocks.abs().amax(dim=(1, 3)).clamp(min=1e-4)
    scale = pow2_ceil(amax / FP8_E4M3_MAX)
    q = (blocks / scale[:, None, :, None]).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    q = q.view(so * bo, si * bi)[:o, :i].to(torch.float8_e4m3fn)
    return q, scale.to(torch.float8_e8m0fnu)


def quant_fp4_group(w: torch.Tensor, group: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D weight to packed e2m1 (int8 [out, in//2]) with ue8m0 scales per ``group`` inputs."""
    o, i = w.shape
    assert i % group == 0 and i % 2 == 0
    g = w.float().view(o, i // group, group)
    amax = g.abs().amax(-1).clamp(min=FP4_MAX * 2.0**-126)
    scale = pow2_ceil(amax / FP4_MAX)
    codes = round_to_fp4((g / scale[..., None]).view(o, i))
    return pack_fp4(codes), scale.to(torch.float8_e8m0fnu)

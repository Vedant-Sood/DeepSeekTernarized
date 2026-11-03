"""Ternary packing formats TQ2 and TQ1.

A ternary weight matrix ``T`` of shape ``[N, K]`` with entries in {-1, 0, +1} is stored row-major
(one output row after another, the reduction dim K contiguous), each trit as the digit
``u = t + 1`` in {0, 1, 2}.

**TQ2** (2 bits/weight): 4 digits per byte, the first in the low bits:
``byte = u0 | u1 << 2 | u2 << 4 | u3 << 6``. A row takes ``ceil(K / 4)`` bytes; padding digits are 1
(t = 0). Unpacking is a shift and a mask, so it is the fast format. The value 3 never occurs.

**TQ1** (1.6 bits/weight): 5 digits per byte in base 3, the first as the least significant digit:
``byte = u0 + 3 u1 + 9 u2 + 27 u3 + 81 u4`` (0..242, since 3^5 = 243 <= 256). A row takes
``ceil(K / 5)`` bytes; padding digits are 1. Unpacking goes through a 243 x 5 lookup table.

Scales are separate: one per output row per group of ``group`` consecutive K entries, shape
``[N, ceil(K / group)]``, FP16 or BF16. A stacked (Mode C) weight is a list of such terms.
"""

from __future__ import annotations

import torch

FORMATS = ("tq2", "tq1")


def bits_per_weight(fmt: str) -> float:
    return {"tq2": 2.0, "tq1": 1.6}[fmt]


def packed_cols(fmt: str, K: int) -> int:
    return -(-K // (4 if fmt == "tq2" else 5))


def _digits(t: torch.Tensor, per: int) -> torch.Tensor:
    N, K = t.shape
    pad = -K % per
    u = (t.to(torch.int16) + 1).to(torch.uint8)
    if pad:
        u = torch.cat([u, torch.ones(N, pad, dtype=torch.uint8, device=t.device)], 1)
    return u.view(N, -1, per)


def pack_tq2(t: torch.Tensor) -> torch.Tensor:
    u = _digits(t, 4).to(torch.int32)
    return (u[..., 0] | (u[..., 1] << 2) | (u[..., 2] << 4) | (u[..., 3] << 6)).to(torch.uint8)


def unpack_tq2(p: torch.Tensor, K: int) -> torch.Tensor:
    b = p.to(torch.int16)
    u = torch.stack([(b >> s) & 3 for s in (0, 2, 4, 6)], -1).flatten(-2)
    return (u[:, :K] - 1).to(torch.int8)


_POW3 = torch.tensor([1, 3, 9, 27, 81], dtype=torch.int32)


def pack_tq1(t: torch.Tensor) -> torch.Tensor:
    u = _digits(t, 5).to(torch.int32)
    return (u * _POW3.to(u.device)).sum(-1).to(torch.uint8)


def tq1_lut() -> torch.Tensor:
    """[256, 5] int8 trits for each byte value (rows >= 243 unused, decoded as 0)."""
    v = torch.arange(256, dtype=torch.int32)
    d = torch.stack([(v // p) % 3 for p in _POW3.tolist()], -1) - 1
    d[243:] = 0
    return d.to(torch.int8)


_TQ1_LUT = tq1_lut()


def unpack_tq1(p: torch.Tensor, K: int) -> torch.Tensor:
    return _TQ1_LUT.to(p.device)[p.long()].flatten(-2)[:, :K]


def pack(t: torch.Tensor, fmt: str) -> torch.Tensor:
    return pack_tq2(t) if fmt == "tq2" else pack_tq1(t)


def unpack(p: torch.Tensor, fmt: str, K: int) -> torch.Tensor:
    return unpack_tq2(p, K) if fmt == "tq2" else unpack_tq1(p, K)

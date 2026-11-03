"""Ternary linear layers (Mode C with k terms, Mode D with k = 1).

Activations into a ternary layer are quantized to int8 per token with an absmax scale (default),
or kept in floating point (``act="bf16"``). The product then runs on the C++ kernel (add/subtract
inner loop) or, without it, on a PyTorch reference that dequantizes the weight.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..kernels.cpu import ext
from .pack import _TQ1_LUT, unpack


def quant_act_int8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token absmax int8: x ~= q * s, q in [-127, 127]."""
    xf = x.float()
    s = (xf.abs().amax(-1, keepdim=True) / 127.0).clamp(min=1e-8)
    q = torch.round(xf / s).clamp(-127, 127).to(torch.int8)
    return q, s.squeeze(-1)


@dataclass
class PackedTerm:
    packed: torch.Tensor  # [N, ceil(K / per_byte)] uint8
    scale: torch.Tensor  # [N, G] fp16/bf16 as stored


class TernaryLinear:
    def __init__(self, terms: list[PackedTerm], K: int, fmt: str, group: int, act: str = "int8", backend: str = "auto"):
        self.terms, self.K, self.fmt, self.group, self.act = terms, K, fmt, group, act
        self.scales_f32 = [t.scale.float().contiguous() for t in terms]
        self.backend = backend
        self._deq = None

    @property
    def out_features(self) -> int:
        return self.terms[0].packed.size(0)

    def dequantized(self) -> torch.Tensor:
        if self._deq is None:
            w = None
            for t, s in zip(self.terms, self.scales_f32):
                d = unpack(t.packed, self.fmt, self.K).float() * s.repeat_interleave(self.group, -1)[:, : self.K]
                w = d if w is None else w + d
            self._deq = w
        return self._deq

    def __call__(self, x: torch.Tensor, out_dtype=torch.bfloat16) -> torch.Tensor:
        shape = x.shape
        x2 = x.reshape(-1, self.K)
        if self.act == "int8":
            q, s = quant_act_int8(x2)
        else:
            q, s = x2.float().contiguous(), torch.ones(x2.size(0))
        e = ext() if self.backend in ("auto", "cpp") else None
        if e is not None and x2.device.type == "cpu":
            out = torch.zeros(x2.size(0), self.out_features, dtype=torch.float32)
            fmt = 2 if self.fmt == "tq2" else 1
            for t, sc in zip(self.terms, self.scales_f32):
                e.ternary_gemm(out, q.contiguous(), s.float().contiguous(), t.packed.contiguous(), sc, self.K, self.group, fmt, _TQ1_LUT)
        else:
            xa = q.float() * s.unsqueeze(-1) if self.act == "int8" else q
            out = xa @ self.dequantized().t()
        return out.view(*shape[:-1], -1).to(out_dtype)

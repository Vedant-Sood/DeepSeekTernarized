"""Absmean ternary linear layers with straight-through gradients.

Scales are per output row and input group, matching the engine's packed format. A short final
group is averaged over its actual entries, not over padding. This module supplies a training
primitive; it does not start a training run or decide which model tensors to replace.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from ..ternary.fit import absmean_ternary
from ..ternary.ops import PackedTerm, quant_act_int8
from ..ternary.pack import pack


def warmup_level(step: int, warmup_steps: int) -> float:
    """Blend fraction at optimizer step ``step``; zero warmup enables quantization immediately."""
    if step < 0 or warmup_steps < 0:
        raise ValueError("step and warmup_steps must be nonnegative")
    return 1.0 if warmup_steps == 0 else min(step / warmup_steps, 1.0)


class BitLinear(nn.Module):
    """Linear layer with BF16 master weights, int8 activation STE, and ternary weight STE."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False, *,
                 group: int = 128, scale_dtype: torch.dtype = torch.float16,
                 master_dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        if in_features <= 0 or out_features <= 0 or group <= 0:
            raise ValueError("feature counts and group must be positive")
        if scale_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("scale_dtype must be float16 or bfloat16")
        self.in_features, self.out_features, self.group = in_features, out_features, group
        self.scale_dtype = scale_dtype
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=master_dtype))
        self.bias = nn.Parameter(torch.empty(out_features, dtype=master_dtype)) if bias else None
        self.level = 0.0
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    @classmethod
    def from_linear(cls, linear: nn.Linear, *, group: int = 128, scale_dtype: torch.dtype = torch.float16,
                    master_dtype: torch.dtype = torch.bfloat16) -> "BitLinear":
        result = cls(linear.in_features, linear.out_features, linear.bias is not None, group=group,
                     scale_dtype=scale_dtype, master_dtype=master_dtype).to(linear.weight.device)
        with torch.no_grad():
            result.weight.copy_(linear.weight)
            if result.bias is not None:
                result.bias.copy_(linear.bias)
        return result

    def set_quantization_level(self, level: float) -> None:
        if not 0.0 <= level <= 1.0:
            raise ValueError("quantization level must be between zero and one")
        self.level = float(level)

    def get_extra_state(self) -> dict:
        return {"quantization_level": self.level}

    def set_extra_state(self, state: dict) -> None:
        self.set_quantization_level(state["quantization_level"])

    @torch.no_grad()
    def packed_term(self, fmt: str) -> PackedTerm:
        """Pack the current master weights in the engine's TQ1/TQ2 layout."""
        if fmt not in ("tq1", "tq2"):
            raise ValueError("packing must be tq1 or tq2")
        term = absmean_ternary(self.weight, self.group, self.scale_dtype)
        return PackedTerm(pack(term.trits, fmt), term.scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.level == 0:
            return F.linear(x, self.weight.to(x.dtype), None if self.bias is None else self.bias.to(x.dtype))
        term = absmean_ternary(self.weight, self.group, self.scale_dtype)
        quant_weight = term.trits.float() * term.scale.float().repeat_interleave(self.group, -1)[:, :self.in_features]
        weight = self.weight.float()
        # The straight-through estimator passes the identity gradient to the master weights.
        weight_ste = quant_weight + (weight - weight.detach())
        xf = x.float()
        qx, sx = quant_act_int8(xf)
        # The absmax scale depends on x; detach the entire quantized value so the only
        # activation gradient is the intended identity estimator.
        quant_x = (qx.float() * sx.unsqueeze(-1)).detach()
        x_ste = quant_x + (xf - xf.detach())
        if self.level == 1:
            weight_blend, x_blend = weight_ste, x_ste
        else:
            weight_blend = torch.lerp(weight, weight_ste, self.level)
            x_blend = torch.lerp(xf, x_ste, self.level)
        bias = None if self.bias is None else self.bias.float()
        return F.linear(x_blend, weight_blend, bias).to(x.dtype)

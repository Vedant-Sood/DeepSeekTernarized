"""Load a Hugging Face Llama/OLMoE checkpoint and prepare it for ternary QAT."""

from __future__ import annotations

import torch
from torch import nn

from .distributed import DEFAULT_PATTERN, convert_linears
from .routing import freeze_routers


class Expert(nn.Module):
    """One SwiGLU expert with separate projections (the published OLMoE tensor layout)."""

    def __init__(self, hidden: int, inter: int, dtype=None):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, inter, bias=False, dtype=dtype)
        self.up_proj = nn.Linear(hidden, inter, bias=False, dtype=dtype)
        self.down_proj = nn.Linear(inter, hidden, bias=False, dtype=dtype)

    def forward(self, x):
        return self.down_proj(nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class Experts(nn.ModuleList):
    """Drop-in for transformers' fused ``OlmoeExperts``: same call signature, one module per expert,
    so each projection can become a BitLinear and be exported/sharded on its own."""

    def forward(self, hidden, top_k_index, top_k_weights):
        out = torch.zeros_like(hidden)
        for e, expert in enumerate(self):
            token, slot = torch.where(top_k_index == e)
            if token.numel():
                out.index_add_(0, token, expert(hidden[token]) * top_k_weights[token, slot, None].to(hidden.dtype))
        return out


def unfuse_experts(model: nn.Module) -> int:
    """Replace fused 3-D expert parameters (recent transformers) by :class:`Experts`; returns layers changed."""
    changed = 0
    for layer in getattr(model.model, "layers", []):
        experts = getattr(getattr(layer, "mlp", None), "experts", None)
        if experts is None or not hasattr(experts, "gate_up_proj"):
            continue
        gate_up, down = experts.gate_up_proj.data, experts.down_proj.data  # [E, 2I, H], [E, H, I]
        n, inter, hidden = gate_up.size(0), gate_up.size(1) // 2, gate_up.size(2)
        new = Experts([Expert(hidden, inter, dtype=gate_up.dtype) for _ in range(n)])
        with torch.no_grad():
            for e, ex in enumerate(new):
                g, u = gate_up[e].chunk(2, 0)
                ex.gate_proj.weight.copy_(g)
                ex.up_proj.weight.copy_(u)
                ex.down_proj.weight.copy_(down[e])
        layer.mlp.experts = new
        changed += 1
    return changed


def load_for_qat(path: str, *, dtype: torch.dtype = torch.bfloat16, group: int = 128,
                 scale_dtype: torch.dtype = torch.float16, pattern: str = DEFAULT_PATTERN,
                 freeze_router: bool = True) -> tuple[nn.Module, dict]:
    """HF model with block linears as BitLinear (master weights in ``dtype``), routers frozen."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype)
    info = {"model_type": model.config.model_type, "unfused_layers": unfuse_experts(model)}
    info["converted"] = convert_linears(model, pattern, group=group, scale_dtype=scale_dtype, master_dtype=dtype)
    info["frozen_routers"] = freeze_routers(model) if freeze_router else 0
    return model, info

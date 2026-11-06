"""Routing-drift diagnostics for MoE models: does ternarization change which experts are picked?

Two numbers per MoE layer, as the M4 plan asks:

* **top-k overlap** with the reference model: for each token, |experts(ref) ∩ experts(candidate)| / k,
  averaged over tokens. 1.0 means identical routing.
* **expert load entropy**: Shannon entropy of the fraction of routed slots each expert receives,
  divided by log(num_experts) so 1.0 is a perfectly balanced load and 0.0 is one expert taking all.

Both models must see the same token ids; each runs on its own hidden states, so the overlap
measures end-to-end drift up to that layer. Sources of routing decisions:

* ``StandardDecoder.forward(..., capture_routing=list)`` for the Boilerplate engine;
* :func:`capture_hf_routing` for a Hugging Face (or QAT training) model, by hooking its router.
"""

from __future__ import annotations

import math
import re
from contextlib import contextmanager

import torch
from torch import nn


def topk_overlap(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    """Mean fraction of shared experts per token between two [tokens, k] index tensors."""
    if reference.shape != candidate.shape or reference.ndim != 2 or reference.numel() == 0:
        raise ValueError("expected two nonempty [tokens, k] tensors of the same shape")
    shared = (reference.unsqueeze(-1) == candidate.unsqueeze(-2)).any(-1).sum(-1)
    return float(shared.float().mean() / reference.size(-1))


def load_entropy(indices: torch.Tensor, num_experts: int) -> float:
    """Normalized entropy (0..1) of how routed slots spread over ``num_experts`` experts."""
    if num_experts < 2 or indices.numel() == 0:
        raise ValueError("need at least two experts and one routed slot")
    counts = torch.bincount(indices.reshape(-1).long(), minlength=num_experts).double()
    if counts.numel() != num_experts:
        raise ValueError("expert index out of range")
    p = counts[counts > 0] / counts.sum()
    return float(-(p * p.log()).sum() / math.log(num_experts))


class RoutingDrift:
    """Accumulates per-layer routing over many batches, then reports overlap and load entropy."""

    def __init__(self, num_layers: int, num_experts: int):
        self.num_experts = num_experts
        self.reference: list[list[torch.Tensor]] = [[] for _ in range(num_layers)]
        self.candidate: list[list[torch.Tensor]] = [[] for _ in range(num_layers)]

    def update(self, reference: list[torch.Tensor], candidate: list[torch.Tensor]) -> None:
        """One entry per MoE layer, each [tokens, k], for the same tokens in both lists."""
        if len(reference) != len(self.reference) or len(candidate) != len(self.candidate):
            raise ValueError("routing must have one entry per MoE layer")
        for store, items in ((self.reference, reference), (self.candidate, candidate)):
            for acc, idx in zip(store, items):
                acc.append(idx.reshape(-1, idx.size(-1)).cpu())

    def summary(self) -> list[dict]:
        out = []
        for layer, (ref, cand) in enumerate(zip(self.reference, self.candidate)):
            if not ref:
                raise ValueError("no routing recorded")
            r, c = torch.cat(ref), torch.cat(cand)
            out.append({
                "layer": layer, "tokens": r.size(0),
                "topk_overlap": topk_overlap(r, c),
                "load_entropy_reference": load_entropy(r, self.num_experts),
                "load_entropy_candidate": load_entropy(c, self.num_experts),
            })
        return out


@contextmanager
def capture_hf_routing(model: nn.Module, top_k: int, pattern: str = r"\.mlp\.gate$"):
    """Record top-k expert ids from every router whose module name matches ``pattern``.

    Yields a list that receives, per forward call, one [tokens, k] tensor per router in model
    order. The router's logits are ranked directly; softmax does not change the top-k set. Works
    on frozen or trainable routers; the hooks only read outputs and are removed on exit.
    """
    routers = [(name, m) for name, m in model.named_modules() if re.search(pattern, name)]
    if not routers:
        raise ValueError(f"no router modules match {pattern!r}")
    calls: list[list[torch.Tensor]] = []
    current: list[torch.Tensor] = []

    def hook(_module, _inputs, output):
        logits = output[0] if isinstance(output, tuple) else output
        current.append(logits.detach().reshape(-1, logits.size(-1)).topk(top_k, dim=-1).indices.cpu())
        if len(current) == len(routers):
            calls.append(list(current))
            current.clear()

    handles = [m.register_forward_hook(hook) for _, m in routers]
    try:
        yield calls
    finally:
        for h in handles:
            h.remove()


def freeze_routers(model: nn.Module, pattern: str = r"\.mlp\.gate$") -> int:
    """Stop gradients into every router (the QAT default); returns how many were frozen."""
    n = 0
    for name, module in model.named_modules():
        if re.search(pattern, name):
            for p in module.parameters():
                p.requires_grad_(False)
            n += 1
    return n

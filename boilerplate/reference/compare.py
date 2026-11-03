"""Compare a candidate trace against a reference trace, layer by layer and on logits."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from .trace import ReferenceTrace


@dataclass
class LayerMetrics:
    index: int
    cos_mean: float  # cosine similarity per token vector, averaged over tokens
    cos_min: float
    rel_l2: float  # ||a - b|| / ||a|| over the whole tensor
    max_abs: float
    bit_exact: bool


@dataclass
class LogitMetrics:
    kl_mean: float  # KL(ref || cand) per token, averaged
    kl_max: float
    top1_agreement: float
    max_abs: float
    bit_exact: bool


@dataclass
class Comparison:
    layers: list[LayerMetrics]
    logits: LogitMetrics | None

    @property
    def min_cos(self) -> float:
        return min((m.cos_min for m in self.layers), default=1.0)

    def to_dict(self) -> dict:
        return asdict(self)

    def table(self) -> str:
        rows = [f"{'layer':>5} {'cos_mean':>10} {'cos_min':>10} {'rel_l2':>10} {'max_abs':>10} exact"]
        for m in self.layers:
            rows.append(f"{m.index:>5} {m.cos_mean:>10.6f} {m.cos_min:>10.6f} {m.rel_l2:>10.2e} {m.max_abs:>10.2e} {m.bit_exact}")
        if self.logits:
            lg = self.logits
            rows.append(f"logits: KL mean {lg.kl_mean:.3e} max {lg.kl_max:.3e}  top1 {lg.top1_agreement:.4f}  max_abs {lg.max_abs:.3e}  exact {lg.bit_exact}")
        return "\n".join(rows)


def _layer_metrics(i: int, a: torch.Tensor, b: torch.Tensor) -> LayerMetrics:
    if a.shape != b.shape:
        raise ValueError(f"hidden state {i}: shape {tuple(a.shape)} vs {tuple(b.shape)}")
    exact = a.dtype == b.dtype and torch.equal(a, b)
    a32, b32 = a.float(), b.float()
    # one vector per token: flatten everything after [batch, seq] (covers [b,s,d] and [b,s,hc,d])
    va, vb = a32.flatten(2), b32.flatten(2)
    cos = F.cosine_similarity(va, vb, dim=-1, eps=1e-12)
    rel = ((a32 - b32).norm() / a32.norm().clamp(min=1e-30)).item()
    return LayerMetrics(i, cos.mean().item(), cos.min().item(), rel, (a32 - b32).abs().max().item(), exact)


def compare_traces(ref: ReferenceTrace, cand: ReferenceTrace) -> Comparison:
    if not torch.equal(ref.input_ids, cand.input_ids):
        raise ValueError("traces were produced from different input_ids")
    if len(ref.hidden_states) != len(cand.hidden_states):
        raise ValueError(f"{len(ref.hidden_states)} vs {len(cand.hidden_states)} hidden states")
    layers = [_layer_metrics(i, a, b) for i, (a, b) in enumerate(zip(ref.hidden_states, cand.hidden_states))]
    logits = None
    if ref.logits is not None and cand.logits is not None:
        lr, lc = ref.logits.float(), cand.logits.float()
        kl = F.kl_div(F.log_softmax(lc, -1), F.log_softmax(lr, -1), log_target=True, reduction="none").sum(-1)
        logits = LogitMetrics(
            kl.mean().item(),
            kl.max().item(),
            (lr.argmax(-1) == lc.argmax(-1)).float().mean().item(),
            (lr - lc).abs().max().item(),
            torch.equal(ref.logits, cand.logits),
        )
    return Comparison(layers, logits)

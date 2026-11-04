"""Mode C: approximate a weight as a sum of k scaled ternary matrices, W ~= sum_i s_i * T_i.

Each term has one scale per output row per group of ``group`` input columns. Terms are fitted
greedily on the residual. For a single term and a single group, the best ternary approximation
``s * T`` of a vector ``r`` (least squares) keeps the j largest-magnitude entries with their signs
and sets ``s`` to their mean magnitude; the error drops by ``(sum of those j magnitudes)^2 / j``, so
the optimal j is found exactly by sorting and scanning a cumulative sum. ``fit_term`` does this for
every (row, group) at once.

Optional GPTQ-style error compensation (``fit_gptq``): columns are quantized left to right and
each column's rounding error is pushed into the columns not yet quantized, weighted by the inverse
Hessian ``H = X^T X`` of calibration inputs, so the *layer output* error is minimized instead of the
weight error. Scales for a group are fitted when the sweep enters it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .pack import pack, unpack


@dataclass
class TernaryTerm:
    trits: torch.Tensor  # [N, K] int8 in {-1, 0, 1}
    scale: torch.Tensor  # [N, G] fp16/bf16


@torch.no_grad()
def absmean_ternary(weight: torch.Tensor, group: int, scale_dtype: torch.dtype = torch.float16) -> TernaryTerm:
    """Absmean ternary quantization per output row and input group, including short groups."""
    if weight.ndim != 2 or group <= 0 or weight.size(1) == 0:
        raise ValueError("weight must be a matrix with a nonempty input dimension and positive group")
    if scale_dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("scale_dtype must be float16 or bfloat16")
    rows, columns = weight.shape
    groups = math.ceil(columns / group)
    data = torch.nn.functional.pad(weight.float(), (0, groups * group - columns)).view(rows, groups, group)
    counts = torch.full((groups,), group, dtype=torch.float32, device=weight.device)
    counts[-1] = columns - (groups - 1) * group
    # Quantize with the scale as stored so QAT and the packed engine agree.
    scale = (data.abs().sum(-1) / counts).to(scale_dtype)
    scale_f32 = scale.float().unsqueeze(-1)
    trits = torch.where(scale_f32 > 0, torch.round(data / scale_f32.clamp_min(1e-30)).clamp(-1, 1), 0)
    return TernaryTerm(trits.to(torch.int8).view(rows, -1)[:, :columns].contiguous(), scale)


def _grouped(x: torch.Tensor, group: int) -> torch.Tensor:
    N, K = x.shape
    pad = -K % group
    if pad:
        x = torch.cat([x, x.new_zeros(N, pad)], 1)
    return x.view(N, -1, group)


def fit_term(r: torch.Tensor, group: int, scale_dtype=torch.float16) -> TernaryTerm:
    """Least-squares optimal single ternary term per (row, group) for residual r [N, K]."""
    N, K = r.shape
    rg = _grouped(r.float(), group)  # [N, G, g]
    mag, order = rg.abs().sort(-1, descending=True)
    csum = mag.cumsum(-1)
    j = torch.arange(1, group + 1, device=r.device, dtype=torch.float32)
    best = (csum.square() / j).argmax(-1, keepdim=True)  # index of the last kept entry
    scale = (csum.gather(-1, best) / (best + 1).float()).squeeze(-1)  # [N, G]
    scale = scale.to(scale_dtype)
    keep_sorted = torch.arange(group, device=r.device) <= best  # [N, G, g] in sorted order
    keep = torch.zeros_like(keep_sorted).scatter_(-1, order, keep_sorted)
    trits = (torch.sign(rg) * keep).to(torch.int8).view(N, -1)[:, :K]
    # an all-zero group gets scale 0; keep it finite and its trits zero
    return TernaryTerm(trits, scale)


def dequant_terms(terms: list[TernaryTerm], K: int, group: int) -> torch.Tensor:
    """sum_i s_i * T_i as an FP32 [N, K] matrix."""
    out = None
    for t in terms:
        s = t.scale.float().repeat_interleave(group, -1)[:, :K]
        w = t.trits.float() * s
        out = w if out is None else out + w
    return out


def fit_residual(w: torch.Tensor, k: int, group: int = 128, scale_dtype=torch.float16) -> list[TernaryTerm]:
    w = w.float()
    terms, r = [], w.clone()
    for _ in range(k):
        t = fit_term(r, group, scale_dtype)
        terms.append(t)
        r -= dequant_terms([t], w.size(1), group)
    return terms


def _quant_column(w: torch.Tensor, scales: list[torch.Tensor]) -> list[torch.Tensor]:
    """Greedy nearest trits for one column given fixed per-row scales of each term."""
    r, out = w.clone(), []
    for s in scales:
        sf = s.float()
        t = torch.where(sf > 0, torch.round(r / sf.clamp(min=1e-30)).clamp(-1, 1), torch.zeros_like(r))
        out.append(t)
        r -= t * sf
    return out


def fit_gptq(w: torch.Tensor, H: torch.Tensor, k: int, group: int = 128, damp: float = 0.01,
             scale_dtype=torch.float16, quantizer: str = "residual") -> list[TernaryTerm]:
    """GPTQ with stacked ternary terms. w: [N, K]; H: [K, K] = X^T X over calibration inputs."""
    if quantizer not in ("residual", "absmean") or (quantizer == "absmean" and k != 1):
        raise ValueError("GPTQ quantizer must be residual or single-term absmean")
    W = w.float().clone()
    N, K = W.shape
    H = H.float().clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    H += damp * torch.diag(H).mean() * torch.eye(K)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    trits = [torch.zeros(N, K, dtype=torch.int8) for _ in range(k)]
    G = -(-K // group)
    scales = [torch.zeros(N, G, dtype=scale_dtype) for _ in range(k)]
    for i1 in range(0, K, group):
        i2 = min(i1 + group, K)
        gi = i1 // group
        # scales for this group, from the (already error-compensated) weights entering it
        gterms = ([absmean_ternary(W[:, i1:i2], group, scale_dtype)] if quantizer == "absmean"
                  else fit_residual(W[:, i1:i2], k, group, scale_dtype))
        for t, gt in enumerate(gterms):
            scales[t][:, gi] = gt.scale[:, 0]
        col_scales = [s[:, gi] for s in scales]
        W1 = W[:, i1:i2].clone()
        Err = torch.zeros_like(W1)
        Hi = Hinv[i1:i2, i1:i2]
        for j in range(i2 - i1):
            w_col = W1[:, j]
            ts = _quant_column(w_col, col_scales)
            q = sum(t * s.float() for t, s in zip(ts, col_scales))
            for t, tt in enumerate(ts):
                trits[t][:, i1 + j] = tt.to(torch.int8)
            e = (w_col - q) / Hi[j, j]
            W1[:, j:] -= e.unsqueeze(1) * Hi[j, j:].unsqueeze(0)
            Err[:, j] = e
        W[:, i2:] -= Err @ Hinv[i1:i2, i2:]
    return [TernaryTerm(t, s) for t, s in zip(trits, scales)]


def roundtrip_check(term: TernaryTerm, fmt: str) -> bool:
    return torch.equal(unpack(pack(term.trits, fmt), fmt, term.trits.size(1)), term.trits)

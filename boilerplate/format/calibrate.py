"""GPTQ calibration: collect H = X^T X of the inputs to selected linears by running the Mode A engine.

``w1`` and ``w3`` of an expert see the same input, so they share one accumulator. Memory is one
FP32 [K, K] matrix per distinct input: 100 MB for a 5120-wide expert input. That is fine for a
truncated model or a few layers, not for all 40 x 384 experts at once; ``layers`` restricts the
capture so a full model can be calibrated in passes.
"""

from __future__ import annotations

import re

import torch

from ..checkpoint.dequant import is_packed_fp4
from ..engine import model as M
from .export import FP4_STORAGE
from .policy import Policy


def _input_key(base: str) -> str:
    return re.sub(r"\.w3$", ".w1", base)


def calibrate(engine, sequences: list[list[int]], policy: Policy, layers: set[int] | None = None, log=print) -> dict[str, torch.Tensor]:
    """{weight base: H / tokens} for every base whose rule asks for GPTQ (optionally only in ``layers``)."""
    ws = engine.model.w
    want = []
    for n in ws.names():
        if not n.endswith(".weight"):
            continue
        b = n[: -len(".weight")]
        r = policy.rule_for(b)
        if r.mode != "C" or not r.gptq or ws.get(n).ndim != 2:
            continue
        m = re.match(r"layers\.(\d+)\.", b)
        if layers is not None and (m is None or int(m.group(1)) not in layers):
            continue
        want.append(b)
    acc: dict[str, list] = {}
    for b in want:
        k = _input_key(b)
        if k not in acc:
            w, sc = ws.get(f"{b}.weight"), ws.maybe(f"{b}.scale")
            fp4 = sc is not None and w.dtype in FP4_STORAGE and is_packed_fp4(tuple(w.shape), tuple(sc.shape))
            K = w.size(1) * (2 if fp4 else 1)
            acc[k] = [torch.zeros(K, K), 0]
    log(f"calibrating {len(want)} weights ({len(acc)} distinct inputs, {sum(a[0].numel() for a in acc.values()) * 4 / 2**30:.1f} GiB of H)")
    M.CALIB = acc
    try:
        cache = engine.new_cache()
        for i, ids in enumerate(sequences):
            q = cache.new_sequence()
            engine.model.forward([(q, ids)], cache)
            cache.free(q)
            log(f"  sequence {i + 1}/{len(sequences)}")
    finally:
        M.CALIB = None
    out = {}
    for b in want:
        H, n = acc[_input_key(b)]
        if n:  # an expert no calibration token routed to gets no H (exported without GPTQ)
            out[b] = H / n
    return out

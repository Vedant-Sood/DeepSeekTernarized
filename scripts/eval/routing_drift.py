"""Routing drift of a converted checkpoint against its source: per-layer top-k overlap and load entropy.

    python scripts/eval/routing_drift.py SOURCE VARIANT TEXT_FILE --ctx 512 --windows 8 --out drift.json

Both checkpoints run the same windows of the tokenized text (prefill only); for every MoE layer the
experts each token picks are compared (``qat/routing.py``: mean |top-k ∩ top-k'| / k, and the
normalized entropy of the expert load in each model). Works for DeepSeek V4.1 (any mode) and
OLMoE. A layer that drifts here routes tokens to different experts than the release does, which
the per-layer cosine alone does not show.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None) -> int:
    import torch

    from boilerplate.engine import Engine
    from boilerplate.qat.routing import RoutingDrift

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("source", type=Path)
    ap.add_argument("variant", type=Path)
    ap.add_argument("text", type=Path)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--windows", type=int, default=8)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    ref = Engine.load(a.source, max_seq_len=a.ctx + 1)
    cand = Engine.load(a.variant, max_seq_len=a.ctx + 1)
    ids = ref.tokenizer.encode(a.text.read_text(encoding="utf-8")).ids
    windows = [ids[i * a.ctx:(i + 1) * a.ctx] for i in range(min(a.windows, len(ids) // a.ctx))]
    if not windows:
        raise SystemExit(f"text has fewer than {a.ctx} tokens")
    drift = None
    for w in windows:
        r, c = ref.routing(w), cand.routing(w)
        if drift is None:
            n_experts = int(max(int(x.max()) for x in r + c)) + 1
            cfg = json.loads((a.source / "config.json").read_text())
            cfg = cfg.get("text_config", cfg)
            n_experts = cfg.get("num_experts") or cfg.get("n_routed_experts") or n_experts
            drift = RoutingDrift(len(r), n_experts)
        drift.update(r, c)
    layers = drift.summary()
    result = {"source": str(a.source), "variant": str(a.variant), "ctx": a.ctx, "windows": len(windows),
              "mean_topk_overlap": sum(x["topk_overlap"] for x in layers) / len(layers),
              "min_topk_overlap": min(x["topk_overlap"] for x in layers), "layers": layers}
    text = json.dumps(result, indent=1)
    if a.out:
        a.out.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

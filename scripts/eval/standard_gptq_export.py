"""Calibrate once, then export several GPTQ policies of a Llama/OLMoE checkpoint.

Hessians H = X^T X / n are collected on the unquantized model (all layers in one pass, not the
sequential quantized-input variant), from ``--seqs`` non-overlapping windows of ``--len`` tokens
of a calibration text. All policies passed share them.

    python scripts/eval/standard_gptq_export.py SRC CALIB_TEXT OUT_ROOT POLICY.yaml [POLICY.yaml ...]
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("calib_text", type=Path)
    p.add_argument("out_root", type=Path)
    p.add_argument("policies", type=Path, nargs="+")
    p.add_argument("--seqs", type=int, default=512)
    p.add_argument("--len", type=int, default=512)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--min-tokens", type=int, default=1)
    a = p.parse_args(argv)
    os.environ.update(OMP_NUM_THREADS=str(a.threads), MKL_NUM_THREADS=str(a.threads), MKL_DYNAMIC="FALSE")

    import hashlib

    import torch

    from boilerplate.engine.standard import StandardEngine
    from boilerplate.format.calibrate import calibrate
    from boilerplate.format.export import export
    from boilerplate.format.policy import Policy

    torch.set_num_threads(a.threads)
    policies = [Policy.load(x) for x in a.policies]
    union = Policy("union", [r for pol in policies for r in pol.rules if r.gptq])
    engine = StandardEngine.load(a.source, max_seq_len=a.len)
    text = a.calib_text.read_text(encoding="utf-8")
    ids = engine.tokenizer.encode(text, add_special_tokens=False).ids
    seqs = [ids[i:i + a.len] for i in range(0, len(ids) - a.len + 1, a.len)][:a.seqs]
    t0 = time.perf_counter()
    hessians = calibrate(engine, seqs, union, min_tokens=a.min_tokens, log=lambda m: print(m, flush=True))
    calib_s = time.perf_counter() - t0
    results = {"calib_text_sha256": hashlib.sha256(text.encode()).hexdigest(), "seqs": len(seqs),
               "len": a.len, "calib_s": round(calib_s, 1), "hessians": len(hessians), "exports": {}}
    for path, pol in zip(a.policies, policies):
        t0 = time.perf_counter()
        out = export(a.source, a.out_root / pol.name, pol, hessians=hessians, log=lambda m: print(m, flush=True))
        meta = json.loads((out / "manifest.json").read_text())
        fallbacks = sum(1 for e in meta["tensors"].values() if e.get("gptq_fallback"))
        results["exports"][pol.name] = {"dir": str(out), "export_s": round(time.perf_counter() - t0, 1),
                                        "gptq_fallbacks": fallbacks, "summary": meta["summary"]}
    print(json.dumps(results, indent=2))
    (a.out_root / "gptq_calibration.json").write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

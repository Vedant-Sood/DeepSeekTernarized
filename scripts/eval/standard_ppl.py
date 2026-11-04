"""WikiText-2 perplexity of a Llama/OLMoE checkpoint (any Boilerplate mode) on the standard decoder.

Protocol (same as llama.cpp ``llama-perplexity`` so the GGUF baselines are comparable): the joined
text (rows joined with blank lines) is tokenized without BOS, cut into non-overlapping windows of
``--ctx`` tokens, and only the second half of each window is scored (each scored token has at
least ctx/2 tokens of context). Perplexity = exp(mean NLL over all scored tokens).

    python scripts/eval/standard_ppl.py CKPT TEXT_FILE --ctx 2048 --threads 4 [--max-windows N]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import time
from pathlib import Path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("text", type=Path, help="UTF-8 text file (already joined)")
    p.add_argument("--ctx", type=int, default=2048)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--max-windows", type=int, help="score only the first N windows (reported)")
    p.add_argument("--ternary-backend", default="torch", choices=("torch", "auto"),
                   help="torch: int8 activations x cached FP32 dequantized trits (BLAS); auto: C++ kernel")
    p.add_argument("--out", type=Path, help="also write the JSON result here")
    a = p.parse_args(argv)
    os.environ.update(OMP_NUM_THREADS=str(a.threads), MKL_NUM_THREADS=str(a.threads), MKL_DYNAMIC="FALSE")

    import torch
    import torch.nn.functional as F

    from boilerplate.engine.standard import StandardEngine

    torch.set_num_threads(a.threads)
    text = a.text.read_text(encoding="utf-8")
    t0 = time.perf_counter()
    engine = StandardEngine.load(a.checkpoint, dtype=torch.float32, max_seq_len=a.ctx,
                                 ternary_backend=a.ternary_backend)
    load_s = time.perf_counter() - t0
    ids = engine.tokenizer.encode(text, add_special_tokens=False).ids
    windows = len(ids) // a.ctx
    if a.max_windows:
        windows = min(windows, a.max_windows)
    first = a.ctx // 2
    nll, count = 0.0, 0
    t0 = time.perf_counter()
    for w in range(windows):
        chunk = torch.tensor([ids[w * a.ctx:(w + 1) * a.ctx]], dtype=torch.long)
        logits, _, _ = engine.model.forward(chunk)
        lp = F.log_softmax(logits[0, first:-1].float(), dim=-1)
        target = chunk[0, first + 1:]
        nll -= float(lp.gather(-1, target[:, None]).sum())
        count += target.numel()
        print(f"window {w + 1}/{windows}: ppl so far {math.exp(nll / count):.4f}", flush=True)
    eval_s = time.perf_counter() - t0
    cpu = platform.processor()
    if Path("/proc/cpuinfo").exists():
        cpu = next((l.split(":", 1)[1].strip() for l in Path("/proc/cpuinfo").read_text().splitlines()
                    if l.startswith("model name")), cpu)
    manifest = a.checkpoint / "manifest.json"
    result = {
        "checkpoint": str(a.checkpoint),
        "policy": json.loads(manifest.read_text())["policy"]["name"] if manifest.exists() else "source (Mode A)",
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(), "tokens_total": len(ids),
        "ctx": a.ctx, "windows": windows, "scored_tokens": count, "nll_sum": nll,
        "ppl": math.exp(nll / count), "compute_dtype": "float32", "ternary_backend": a.ternary_backend, "threads": torch.get_num_threads(),
        "cpu": cpu, "torch": torch.__version__, "load_s": round(load_s, 2), "eval_s": round(eval_s, 1),
    }
    print(json.dumps(result, indent=2))
    if a.out:
        a.out.write_text(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

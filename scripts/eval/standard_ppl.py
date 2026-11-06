"""WikiText-2 perplexity of a checkpoint (Llama/OLMoE on the standard decoder, DeepSeek V4.1 on the main engine; any mode).

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
    p.add_argument("--ternary-backend", default="torch", choices=("torch", "auto", "cuda"),
                   help="torch: int8 activations x cached FP32 dequantized trits (BLAS); auto: C++ kernel")
    p.add_argument("--out", type=Path, help="also write the JSON result here")
    p.add_argument("--device", help="Llama/OLMoE: run on this device (e.g. cuda)")
    a = p.parse_args(argv)
    os.environ.update(OMP_NUM_THREADS=str(a.threads), MKL_NUM_THREADS=str(a.threads), MKL_DYNAMIC="FALSE")

    import torch
    import torch.nn.functional as F

    from boilerplate.engine import Engine
    from boilerplate.engine.standard import StandardEngine

    torch.set_num_threads(a.threads)
    text = a.text.read_text(encoding="utf-8")
    t0 = time.perf_counter()
    model_type = json.loads((a.checkpoint / "config.json").read_text()).get("model_type")
    if model_type in ("llama", "olmoe"):
        engine = StandardEngine.load(a.checkpoint, dtype=torch.float32, max_seq_len=a.ctx,
                                     ternary_backend=a.ternary_backend, device=a.device)
        def window_logits(chunk):
            return engine.model.forward(chunk)[0]
    else:  # DeepSeek V4.1 (any mode) through the main engine
        engine = Engine.load(a.checkpoint, max_seq_len=a.ctx)
        def window_logits(chunk):
            return engine.trace(chunk).logits
    load_s = time.perf_counter() - t0
    ids = engine.tokenizer.encode(text, add_special_tokens=False).ids
    windows = len(ids) // a.ctx
    if a.max_windows:
        windows = min(windows, a.max_windows)
    first = a.ctx // 2
    nll, count, start, prior_s = 0.0, 0, 0, 0.0
    if a.out and a.out.exists():  # resume a partial run with the same settings
        prev = json.loads(a.out.read_text())
        if prev.get("ctx") == a.ctx and prev.get("tokens_total") == len(ids) and prev.get("windows_done", 0) <= windows:
            nll, count, start, prior_s = prev["nll_sum"], prev["scored_tokens"], prev["windows_done"], prev.get("eval_s", 0.0)
            print(f"resuming after {start} windows", flush=True)
    t0 = time.perf_counter()
    for w in range(start, windows):
        chunk = torch.tensor([ids[w * a.ctx:(w + 1) * a.ctx]], dtype=torch.long)
        logits = window_logits(chunk)
        lp = F.log_softmax(logits[0, first:-1].float(), dim=-1)
        target = chunk[0, first + 1:]
        nll -= float(lp.gather(-1, target[:, None]).sum())
        count += target.numel()
        print(f"window {w + 1}/{windows}: ppl so far {math.exp(nll / count):.4f}", flush=True)
        if a.out:
            a.out.write_text(json.dumps({"ctx": a.ctx, "tokens_total": len(ids), "windows_done": w + 1,
                                         "nll_sum": nll, "scored_tokens": count,
                                         "eval_s": prior_s + time.perf_counter() - t0, "partial": True}))
    eval_s = prior_s + time.perf_counter() - t0
    cpu = platform.processor()
    if Path("/proc/cpuinfo").exists():
        cpu = next((l.split(":", 1)[1].strip() for l in Path("/proc/cpuinfo").read_text().splitlines()
                    if l.startswith("model name")), cpu)
    manifest = a.checkpoint / "manifest.json"
    result = {
        "checkpoint": str(a.checkpoint),
        "policy": json.loads(manifest.read_text())["policy"]["name"] if manifest.exists() else "source (Mode A)",
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(), "tokens_total": len(ids),
        "ctx": a.ctx, "windows": windows, "windows_done": windows, "windows_available": len(ids) // a.ctx, "scored_tokens": count, "nll_sum": nll,
        "ppl": math.exp(nll / count), "compute_dtype": "float32", "ternary_backend": a.ternary_backend, "threads": torch.get_num_threads(),
        "cpu": cpu, "torch": torch.__version__, "load_s": round(load_s, 2), "eval_s": round(eval_s, 1),
    }
    print(json.dumps(result, indent=2))
    if a.out:
        a.out.write_text(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Check exact Mode A/Mode B traces on the same tokens and CPU math settings."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def same_bits(a, b) -> bool:
    """Compare representation bytes, including signed zero and NaN payloads."""
    import torch

    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="local Mode A checkpoint directory")
    parser.add_argument("mode_b", type=Path, help="local Mode B export directory")
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--decode-from", type=int, default=248)
    parser.add_argument("--input-ids", type=Path, help="optional safetensors file with input_ids [batch, sequence]")
    parser.add_argument("--seed", type=int, default=0, help="seed when generating input IDs")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--budgets-mib", type=int, nargs="+", default=[0, 256])
    args = parser.parse_args(argv)
    if args.tokens <= 0 or not 0 < args.decode_from <= args.tokens or args.threads <= 0:
        parser.error("tokens and threads must be positive; decode-from must be in 1..tokens")
    if any(budget < 0 for budget in args.budgets_mib):
        parser.error("cache budgets must be nonnegative")
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_DYNAMIC"] = "FALSE"
    os.environ["OMP_DYNAMIC"] = "FALSE"

    import torch
    from safetensors.torch import load_file

    from boilerplate.engine import Engine
    from boilerplate.engine.config import EngineConfig

    torch.set_num_threads(args.threads)
    source_manifest = args.source / "manifest.json"
    if source_manifest.exists():
        source_meta = json.loads(source_manifest.read_text())
        if any(entry["mode"] != "A" for entry in source_meta["tensors"].values()):
            parser.error("source must be a Mode A checkpoint")
    manifest = json.loads((args.mode_b / "manifest.json").read_text())
    if not any(entry["mode"] == "B" and entry.get("coded") for entry in manifest["tensors"].values()):
        parser.error("mode_b must contain at least one entropy-coded Mode B tensor")
    cfg = EngineConfig.from_dir(args.source)
    if args.input_ids is None:
        ids = torch.randint(0, cfg.vocab_size, (1, args.tokens), generator=torch.Generator().manual_seed(args.seed))
    else:
        ids = load_file(str(args.input_ids))["input_ids"]
        if ids.ndim != 2 or ids.shape[1] < args.tokens or ids.dtype not in (torch.int32, torch.int64):
            parser.error("input_ids must be integer [batch, sequence] with at least --tokens positions")
        ids = ids[:, :args.tokens].long()
        if ids.numel() == 0 or int(ids.min()) < 0 or int(ids.max()) >= cfg.vocab_size:
            parser.error("input_ids must be in the source checkpoint's vocabulary")

    def load(path, budget):
        return Engine.load(path, max_seq_len=args.tokens + 8, max_seqs=ids.shape[0],
                           max_tokens=ids.shape[0] * (args.tokens + 64), mode_b_cache_bytes=budget)

    print("Tracing the Mode A source...", file=sys.stderr, flush=True)
    baseline_engine = load(args.source, 0)
    baseline = baseline_engine.trace(ids, decode_from=args.decode_from)
    del baseline_engine
    results = []
    for budget in args.budgets_mib:
        print(f"Tracing Mode B with a {budget} MiB decoded-cache budget...", file=sys.stderr, flush=True)
        engine = load(args.mode_b, budget << 20)
        candidate = engine.trace(ids, decode_from=args.decode_from)
        if len(baseline.hidden_states) != len(candidate.hidden_states):
            raise ValueError("Mode A and Mode B returned different layer counts")
        layer_equal = [same_bits(a, b) for a, b in zip(baseline.hidden_states, candidate.hidden_states)]
        logits_equal = baseline.logits is not None and candidate.logits is not None and same_bits(baseline.logits, candidate.logits)
        results.append({
            "cache_budget_bytes": budget << 20,
            "cache_used_bytes": engine.model.w.mode_b_cache_used,
            "cache_hits": engine.model.w.mode_b_cache_hits,
            "cache_misses": engine.model.w.mode_b_cache_misses,
            "cache_evictions": engine.model.w.mode_b_cache_evictions,
            "layer_bytes_identical": layer_equal,
            "logits_bytes_identical": logits_equal,
        })
        print(f"Completed {budget} MiB: layers={layer_equal}, logits={logits_equal}", file=sys.stderr, flush=True)
        del engine, candidate
    print(json.dumps({
        "source": str(args.source), "mode_b": str(args.mode_b), "batch": ids.shape[0],
        "tokens": args.tokens, "decode_from": args.decode_from,
        "input_ids": str(args.input_ids) if args.input_ids else None,
        "seed": None if args.input_ids else args.seed, "torch": torch.__version__,
        "threads": torch.get_num_threads(), "omp_num_threads": os.environ["OMP_NUM_THREADS"],
        "mkl_num_threads": os.environ["MKL_NUM_THREADS"], "mkl_dynamic": os.environ["MKL_DYNAMIC"],
        "omp_dynamic": os.environ["OMP_DYNAMIC"], "results": results,
    }, indent=2))
    return 0 if all(r["logits_bytes_identical"] and all(r["layer_bytes_identical"]) for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

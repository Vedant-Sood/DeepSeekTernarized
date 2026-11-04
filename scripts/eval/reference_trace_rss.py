"""Trace a local checkpoint with the official reference or the engine, recording peak RSS."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=("official", "engine"))
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("input_ids", type=Path)
    parser.add_argument("output", type=Path, help="new trace path prefix")
    parser.add_argument("--decode-from", type=int, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--converted-dir", type=Path, help="preconverted reference checkpoint directory")
    parser.add_argument("--code-dir", type=Path, default=Path("third_party/deepseek_v41_inference"))
    args = parser.parse_args(argv)
    if args.threads <= 0:
        parser.error("threads must be positive")
    if args.backend == "official" and (args.converted_dir is None or
                                      not (args.converted_dir / "model0-mp1.safetensors").is_file()):
        parser.error("official tracing requires a preconverted --converted-dir")
    if args.output.with_suffix(".safetensors").exists() or args.output.with_suffix(".json").exists():
        parser.error("output trace already exists")
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_DYNAMIC"] = "FALSE"
    os.environ["OMP_DYNAMIC"] = "FALSE"

    import resource
    import torch
    from safetensors.torch import load_file

    from boilerplate.cli import _rss_bytes
    from boilerplate.engine import Engine
    from boilerplate.reference.official_backend import OfficialReference, load_hf_tokenizer

    torch.set_num_threads(args.threads)
    ids = load_file(str(args.input_ids))["input_ids"]
    if ids.ndim != 2 or ids.dtype not in (torch.int32, torch.int64) or not 0 < args.decode_from <= ids.shape[1]:
        parser.error("input_ids must be integer [batch, sequence] and decode-from must be in 1..sequence")
    ids = ids.long()
    t0 = time.perf_counter()
    if args.backend == "official":
        tokenizer = load_hf_tokenizer(args.checkpoint)
        runner = OfficialReference(args.code_dir, args.checkpoint, args.converted_dir,
                                   max_seq_len=ids.shape[1] + 8, max_batch_size=ids.shape[0], tokenizer=tokenizer)
    else:
        runner = Engine.load(args.checkpoint, max_seq_len=ids.shape[1] + 8, max_seqs=ids.shape[0],
                             max_tokens=ids.shape[0] * (ids.shape[1] + 64))
    trace = runner.trace(ids, decode_from=args.decode_from)
    elapsed = time.perf_counter() - t0
    peak = _rss_bytes(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, sys.platform)
    if torch.get_num_threads() != args.threads:
        raise RuntimeError("PyTorch thread count changed during the measurement")
    trace.save(args.output)
    cpu = platform.processor()
    if Path('/proc/cpuinfo').exists():
        for line in Path('/proc/cpuinfo').read_text().splitlines():
            if line.startswith('model name'):
                cpu = line.split(':', 1)[1].strip()
                break
    print(json.dumps({
        "backend": args.backend, "checkpoint": str(args.checkpoint),
        "input_ids_sha256": hashlib.sha256(args.input_ids.read_bytes()).hexdigest(),
        "batch": ids.shape[0], "tokens": ids.shape[1], "decode_from": args.decode_from,
        "cpu": cpu, "threads": torch.get_num_threads(), "omp_num_threads": os.environ["OMP_NUM_THREADS"],
        "mkl_dynamic": os.environ["MKL_DYNAMIC"], "torch": torch.__version__,
        "elapsed_seconds": elapsed, "peak_rss_bytes": peak, "peak_rss_gib": peak / (1 << 30),
        "scope": "model load and trace; reference conversion and trace-file writing excluded",
        "trace": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

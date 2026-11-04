"""Resumable standard-decoder perplexity on fixed, already-tokenized text.

Each non-overlapping context scores logits in its second half, matching llama.cpp's default
perplexity window convention. The input safetensors file must contain a 1-D ``input_ids`` tensor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import resource
import sys
import tempfile
from pathlib import Path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _write_atomic(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("input_ids", type=Path)
    parser.add_argument("--state", type=Path, required=True, help="checkpointed progress and final result JSON")
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-chunks", type=int, help="optional limit for a separately labelled smoke run")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.context < 4 or args.threads <= 0 or (args.max_chunks is not None and args.max_chunks <= 0):
        parser.error("context must be at least four; threads and max-chunks must be positive")
    os.environ.update(OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads),
                      MKL_DYNAMIC="FALSE", OMP_DYNAMIC="FALSE")

    import torch
    from safetensors.torch import load_file
    from boilerplate.cli import _rss_bytes
    from boilerplate.engine import Engine
    from boilerplate.engine.standard import StandardEngine

    torch.set_num_threads(args.threads)
    ids = load_file(str(args.input_ids))["input_ids"]
    if ids.ndim != 1 or ids.dtype not in (torch.int32, torch.int64) or ids.numel() < 2 * args.context:
        parser.error("input_ids must be a 1-D integer tensor with at least two complete contexts")
    ids = ids.long()
    n_chunks = ids.numel() // args.context
    cpu = platform.processor()
    if Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    model_files = sorted(p for p in args.checkpoint.iterdir()
                         if p.is_file() and (p.suffix == ".safetensors" or p.name in {"config.json", "manifest.json", "model.safetensors.index.json"}))
    if not model_files:
        parser.error("checkpoint has no model files")
    identity = {"checkpoint": str(args.checkpoint.resolve()), "model_sha256": {p.name: _sha256(p) for p in model_files},
                "input_ids_sha256": _sha256(args.input_ids), "context": args.context, "threads": args.threads,
                "cpu": cpu, "torch": torch.__version__}
    if args.resume:
        state = json.loads(args.state.read_text())
        if state["identity"] != identity or state["total_chunks"] != n_chunks:
            raise ValueError("checkpoint, input, context or measurement environment changed")
    else:
        if args.state.exists():
            parser.error("state already exists; use --resume or a new path")
        state = {"identity": identity, "total_chunks": n_chunks, "completed_chunks": 0,
                 "scored_tokens": 0, "nll_sum": 0.0, "nll_squared_sum": 0.0,
                 "peak_rss_bytes": 0, "runs": 0}
    state["runs"] += 1
    engine = Engine.load(args.checkpoint, max_seq_len=args.context + 1, max_seqs=1, max_tokens=args.context + 64)
    if not isinstance(engine, StandardEngine):
        raise ValueError("perplexity script requires a standard decoder checkpoint")
    stop = n_chunks if args.max_chunks is None else min(n_chunks, state["completed_chunks"] + args.max_chunks)
    with torch.inference_mode():
        for chunk in range(state["completed_chunks"], stop):
            batch = ids[chunk * args.context:(chunk + 1) * args.context].unsqueeze(0)
            logits, hidden, cache = engine.model.forward(batch)
            del hidden, cache
            first = args.context // 2
            selected = logits[:, first:-1, :].float()
            labels = batch[:, first + 1:]
            nll = torch.logsumexp(selected, dim=-1) - selected.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
            if not torch.isfinite(nll).all():
                raise RuntimeError(f"non-finite NLL in chunk {chunk}")
            nll = nll.double()
            state["nll_sum"] += float(nll.sum())
            state["nll_squared_sum"] += float(nll.square().sum())
            state["scored_tokens"] += nll.numel()
            state["completed_chunks"] = chunk + 1
            state["peak_rss_bytes"] = max(state["peak_rss_bytes"], _rss_bytes(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, sys.platform))
            state["complete"] = state["completed_chunks"] == n_chunks
            state["perplexity"] = math.exp(state["nll_sum"] / state["scored_tokens"])
            if state["scored_tokens"] > 1:
                mean = state["nll_sum"] / state["scored_tokens"]
                variance = max(0.0, state["nll_squared_sum"] / state["scored_tokens"] - mean * mean)
                state["uncertainty"] = state["perplexity"] * math.sqrt(variance / (state["scored_tokens"] - 1))
            state["peak_rss_gib"] = state["peak_rss_bytes"] / (1 << 30)
            _write_atomic(args.state, state)
            print(f"{state['completed_chunks']}/{n_chunks} chunks; partial PPL {state['perplexity']:.6f}", file=sys.stderr, flush=True)
    print(json.dumps(state, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

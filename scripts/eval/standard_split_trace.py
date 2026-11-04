"""Record and compare standard-decoder traces one prompt at a time.

The HF and Boilerplate processes are separate so a large source checkpoint need not coexist
with a second copy. Completed prompts are written atomically and can be resumed.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import resource
import sys
import tempfile
import time
from pathlib import Path


def _hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _cpu() -> str:
    path = Path("/proc/cpuinfo")
    if path.exists():
        for line in path.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor()


def _write_json(path: Path, value: dict) -> None:
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def record(args):
    import torch
    import transformers
    from safetensors.torch import load_file, save_file
    from boilerplate.cli import _rss_bytes
    from boilerplate.engine import Engine
    from boilerplate.engine.standard import StandardEngine

    torch.set_num_threads(args.threads)
    ids = load_file(str(args.input_ids))["input_ids"]
    if ids.ndim != 2 or ids.dtype not in (torch.int32, torch.int64) or not ids.shape[0] or not ids.shape[1]:
        raise ValueError("input_ids must be a nonempty integer [prompts, context] tensor")
    ids = ids.long()
    if args.start < 0 or args.start >= ids.shape[0] or args.count <= 0:
        raise ValueError("start and count do not select any prompts")
    end = min(ids.shape[0], args.start + args.count)
    args.out.mkdir(parents=True, exist_ok=True)
    metadata = args.out / "run.json"
    files = sorted(p for p in args.checkpoint.iterdir() if p.is_file() and
                   (p.suffix == ".safetensors" or p.name in {"config.json", "manifest.json", "model.safetensors.index.json"}))
    identity = {"backend": args.backend, "checkpoint": str(args.checkpoint.resolve()),
                "model_sha256": {p.name: _hash(p) for p in files}, "input_ids_sha256": _hash(args.input_ids),
                "tokens": args.tokens, "prompts": ids.shape[0], "threads": args.threads, "cpu": _cpu(), "torch": torch.__version__,
                "transformers": transformers.__version__}
    if metadata.exists():
        if not args.resume:
            raise ValueError("trace directory already contains a run; pass --resume")
        state = json.loads(metadata.read_text())
        if state["identity"] != identity:
            raise ValueError("checkpoint, input or measurement environment changed")
    else:
        state = {"identity": identity, "completed": {}, "runs": 0}
    state["runs"] += 1
    _write_json(metadata, state)
    started = time.perf_counter()
    if args.backend == "hf":
        model = transformers.AutoModelForCausalLM.from_pretrained(args.checkpoint, dtype=torch.float32,
                                                                  local_files_only=True, attn_implementation="eager").eval()
        reference_layers = []
        def capture(_module, _inputs, output):
            reference_layers.append(output[0] if isinstance(output, tuple) else output)
        hooks = [layer.register_forward_hook(capture) for layer in model.model.layers]
    else:
        model = Engine.load(args.checkpoint, dtype=torch.float32, max_seq_len=ids.shape[1] + args.tokens + 1,
                            max_seqs=1, max_tokens=ids.shape[1] + args.tokens + 64)
        if not isinstance(model, StandardEngine):
            raise ValueError("engine backend requires a standard decoder")
        hooks = []
    try:
        with torch.inference_mode():
            for index in range(args.start, end):
                destination = args.out / f"prompt-{index:03d}.safetensors"
                if destination.exists():
                    if not args.resume:
                        raise ValueError(f"trace already exists: {destination}")
                    digest = _hash(destination)
                    previous = state["completed"].get(str(index))
                    if previous is not None and digest != previous["sha256"]:
                        raise ValueError(f"trace checksum mismatch: {destination}")
                    if previous is None:
                        # A restart can occur after the atomic file rename but before run.json is
                        # updated. Validate that orphan against this run before recovering it.
                        existing = load_file(str(destination))
                        n_layers = int(json.loads((args.checkpoint / "config.json").read_text())["num_hidden_layers"])
                        keys = [f"layer.{i}" for i in range(n_layers)]
                        if ("input_ids" not in existing or not torch.equal(existing["input_ids"], ids[index:index + 1]) or
                                tuple(existing.get("generated", torch.empty(0)).shape) != (1, args.tokens) or
                                any(key not in existing or existing[key].shape[1] != ids.shape[1] + args.tokens - 1 for key in keys)):
                            raise ValueError(f"orphan trace does not match this run: {destination}")
                        state["completed"][str(index)] = {"file": destination.name, "sha256": digest,
                                                         "recovered_after_interruption": True}
                        _write_json(metadata, state)
                    continue
                prompt_started = time.perf_counter()
                current, past = ids[index:index + 1], None
                captured, generated = None, []
                for step in range(args.tokens):
                    if args.backend == "hf":
                        reference_layers.clear()
                        result = model(current, past_key_values=past, use_cache=True)
                        logits, past, layers = result.logits, result.past_key_values, reference_layers
                    else:
                        layers = []
                        logits, _, past = model.model.forward(current, past, capture_layers=layers)
                    if captured is None:
                        captured = [[] for _ in layers]
                    if len(captured) != len(layers):
                        raise RuntimeError("decoder layer count changed")
                    if not torch.isfinite(logits).all() or any(not torch.isfinite(layer).all() for layer in layers):
                        raise RuntimeError(f"non-finite output at prompt {index}, step {step}")
                    for acc, layer in zip(captured, layers):
                        acc.append(layer.detach().cpu().clone())
                    token = logits[:, -1].argmax(-1)
                    generated.append(token.cpu())
                    current = token[:, None]
                    if (step + 1) % 8 == 0 or step + 1 == args.tokens:
                        print(f"{args.backend} prompt {index}: {step + 1}/{args.tokens} tokens", file=sys.stderr, flush=True)
                tensors = {"input_ids": ids[index:index + 1], "generated": torch.stack(generated, dim=1)}
                tensors.update({f"layer.{i}": torch.cat(parts, dim=1) for i, parts in enumerate(captured)})
                fd, temp = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=args.out)
                os.close(fd)
                try:
                    save_file(tensors, temp)
                    os.replace(temp, destination)
                finally:
                    if os.path.exists(temp):
                        os.unlink(temp)
                peak = _rss_bytes(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, sys.platform)
                state["completed"][str(index)] = {"file": destination.name, "sha256": _hash(destination),
                                                 "elapsed_seconds": time.perf_counter() - prompt_started,
                                                 "peak_rss_bytes": peak, "peak_rss_gib": peak / (1 << 30)}
                state["elapsed_seconds_this_run"] = time.perf_counter() - started
                _write_json(metadata, state)
                del past, captured, generated, tensors, logits, layers, current
                if args.backend == "hf":
                    reference_layers.clear()
                    del result
                gc.collect()
    finally:
        for hook in hooks:
            hook.remove()
    print(json.dumps(state, indent=2))
    return 0


def compare(args):
    import torch
    import torch.nn.functional as F
    from safetensors.torch import load_file

    torch.set_num_threads(args.threads)
    ref_info = json.loads((args.reference / "run.json").read_text())
    cand_info = json.loads((args.candidate / "run.json").read_text())
    ref_identity, cand_identity = ref_info["identity"], cand_info["identity"]
    for key in ("model_sha256", "input_ids_sha256", "tokens", "prompts", "cpu", "threads", "torch"):
        if ref_identity[key] != cand_identity[key]:
            raise ValueError(f"reference and candidate differ in {key}")
    tokens = ref_identity["tokens"]
    common = sorted(set(ref_info["completed"]) & set(cand_info["completed"]), key=int)
    if not common:
        raise ValueError("no completed prompts to compare")
    layer_min, prompt_results = None, []
    matching_tokens = 0
    for index in common:
        left = ref_info["completed"][index]
        right = cand_info["completed"][index]
        ref_path, cand_path = args.reference / left["file"], args.candidate / right["file"]
        if _hash(ref_path) != left["sha256"] or _hash(cand_path) != right["sha256"]:
            raise ValueError(f"trace checksum mismatch for prompt {index}")
        ref, cand = load_file(str(ref_path)), load_file(str(cand_path))
        if not torch.equal(ref["input_ids"], cand["input_ids"]):
            raise ValueError(f"prompt {index} inputs differ")
        if ref["generated"].shape != (1, tokens) or cand["generated"].shape != (1, tokens):
            raise ValueError(f"prompt {index} has an invalid generated-token shape")
        match = ref["generated"] == cand["generated"]
        matching_tokens += int(match.sum())
        mismatch = (~match).nonzero(as_tuple=False)
        first_mismatch = None if mismatch.numel() == 0 else int(mismatch[0, 1])
        # The logits at a mismatching step still came from an identical prefix.
        positions = ref["input_ids"].shape[1] + (tokens if first_mismatch is None else first_mismatch)
        keys = sorted((k for k in ref if k.startswith("layer.")), key=lambda k: int(k.split(".")[1]))
        if not keys or keys != sorted((k for k in cand if k.startswith("layer.")), key=lambda k: int(k.split(".")[1])):
            raise ValueError("layer counts differ")
        if layer_min is None:
            layer_min = [float("inf")] * len(keys)
        for layer, key in enumerate(keys):
            a, b = ref[key][:, :positions].float(), cand[key][:, :positions].float()
            if a.shape != b.shape:
                raise ValueError(f"layer shape differs for prompt {index}")
            cosine = F.cosine_similarity(a, b, dim=-1)
            if not torch.isfinite(cosine).all():
                raise ValueError(f"non-finite cosine for prompt {index}, layer {layer}")
            layer_min[layer] = min(layer_min[layer], float(cosine.min()))
        prompt_results.append({"prompt": int(index), "identical_tokens": first_mismatch is None,
                               "first_mismatch": first_mismatch})
    complete = len(common) == ref_identity["prompts"]
    result = {"prompts_compared": len(common), "expected_prompts": ref_identity["prompts"],
              "complete": complete, "tokens_per_prompt": tokens,
              "matching_greedy_tokens": matching_tokens, "total_greedy_tokens": len(common) * tokens,
              "prompts_with_identical_greedy_tokens": sum(r["identical_tokens"] for r in prompt_results),
              "layer_min_cosine_same_prefix": layer_min, "prompts": prompt_results}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        _write_json(args.out, result)
    print(json.dumps(result, indent=2))
    return 0 if (complete or args.allow_partial) and all(r["identical_tokens"] for r in prompt_results) and min(layer_min) >= 0.9999 else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("record")
    run.add_argument("backend", choices=("hf", "engine"))
    run.add_argument("checkpoint", type=Path)
    run.add_argument("input_ids", type=Path)
    run.add_argument("out", type=Path)
    run.add_argument("--tokens", type=int, default=64)
    run.add_argument("--threads", type=int, default=4)
    run.add_argument("--start", type=int, default=0)
    run.add_argument("--count", type=int, default=20)
    run.add_argument("--resume", action="store_true")
    check = sub.add_parser("compare")
    check.add_argument("reference", type=Path)
    check.add_argument("candidate", type=Path)
    check.add_argument("--threads", type=int, default=4)
    check.add_argument("--out", type=Path)
    check.add_argument("--allow-partial", action="store_true", help="allow a labelled subset comparison")
    args = parser.parse_args(argv)
    if (args.command == "record" and args.tokens <= 0) or args.threads <= 0:
        parser.error("tokens and threads must be positive")
    os.environ.update(OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads),
                      MKL_DYNAMIC="FALSE", OMP_DYNAMIC="FALSE", HF_HUB_OFFLINE="1")
    return record(args) if args.command == "record" else compare(args)


if __name__ == "__main__":
    raise SystemExit(main())

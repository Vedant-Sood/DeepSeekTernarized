"""Command-line entry point: ``boilerplate <command>``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def _rss_bytes(ru_maxrss: int, platform: str) -> int:
    """Convert getrusage's peak RSS to bytes on supported platforms."""
    if platform.startswith("linux"):
        return ru_maxrss * 1024  # KiB on Linux
    if platform == "darwin":
        return ru_maxrss  # bytes on macOS
    raise ValueError(f"unsupported ru_maxrss units on platform {platform!r}")


def _cmd_inspect(a) -> int:
    from .checkpoint import CheckpointSource
    from .config import parse_checkpoint
    from .config.report import format_breakdown, size_breakdown

    src = CheckpointSource.open(a.checkpoint, read_headers=not a.no_headers, revision=a.revision)
    spec = parse_checkpoint(src)
    d = spec.to_dict()
    if a.json:
        with open(a.json, "w") as f:
            json.dump(d, f, indent=2, default=str)
    att, moe = spec.attention, spec.moe
    print(f"{spec.architecture} ({spec.model_type}, family {spec.family}) from {spec.source}")
    print(f"layers {spec.num_layers}  hidden {spec.hidden_size}  vocab {spec.vocab_size}  norm_eps {spec.norm_eps}")
    print(f"attention {att.type}: heads {att.num_heads} kv_heads {att.num_kv_heads} head_dim {att.head_dim} rope_dim {att.qk_rope_head_dim}")
    kinds = {}
    for l in att.layers:
        kinds.setdefault((l.kind, l.compress_ratio), []).append(l.layer)
    for (k, r), ls in kinds.items():
        print(f"  {k}{f' r={r}' if r else ''}: layers {ls}")
    if moe:
        print(
            f"moe: {moe.n_routed_experts} routed + {moe.n_shared_experts} shared, top-{moe.top_k}, inter {moe.moe_intermediate_size}, "
            f"score {moe.scoring_func}, norm_topk {moe.norm_topk_prob}, scale {moe.routed_scaling_factor}, bias {moe.router_bias}"
        )
    print(f"rope: {spec.rope}")
    if spec.rope_compress:
        print(f"rope (compressed layers): {spec.rope_compress}")
    for k in ("hyper_connections", "engram", "mtp"):
        if d.get(k):
            print(f"{k}: {json.dumps(d[k], default=str)[:300]}")
    print(f"storage (verified from headers: {spec.storage.verified_from_headers}):")
    for role, fm in sorted(spec.storage.role_formats.items()):
        print(f"  {role:<20} {fm}")
    if src.headers is not None:
        print(format_breakdown(size_breakdown(src)))
    for w in spec.warnings:
        print(f"WARNING: {w}")
    return 0


def _cmd_make_tiny(a) -> int:
    from .testing.tiny import write_tiny_checkpoint

    print(write_tiny_checkpoint(a.out_dir, seed=a.seed))
    return 0


def _cmd_make_truncated(a) -> int:
    from .testing.truncate import make_truncated

    layers = [int(x) for x in a.layers.split(",")]
    print(make_truncated(a.source, a.out_dir, layers, a.experts, keep_engram=a.keep_engram, revision=a.revision))
    return 0


def _input_ids(a, vocab: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(a.seed)
    return torch.randint(0, vocab, (a.batch, a.seq_len), generator=g)


def _cmd_reference(a) -> int:
    if a.backend == "official":
        from .reference.official_backend import OfficialReference

        from .reference.official_backend import load_hf_tokenizer

        ref = OfficialReference(
            a.code_dir, a.checkpoint, a.work_dir or f"{a.out}.converted", use_cpu_kernels=not a.gpu_kernels,
            text_only=a.text_only, max_seq_len=max(a.seq_len, 64), max_batch_size=a.batch,
            tokenizer=load_hf_tokenizer(a.tokenizer or a.checkpoint),
        )
        vocab = ref.args_kwargs["vocab_size"]
    else:
        from .reference.hf_backend import HFReference

        ref = HFReference(a.checkpoint)
        vocab = ref.model.config.vocab_size
    ids = _input_ids(a, vocab)
    if getattr(a, "text", None):
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(str(__import__("pathlib").Path(a.tokenizer or a.checkpoint) / "tokenizer.json"))
        ids = torch.tensor([tok.encode(a.text).ids] * a.batch)
    trace = ref.trace(ids, decode_from=a.decode_from) if a.backend == "official" else ref.trace(ids)
    trace.save(a.out)
    print(f"saved {len(trace.hidden_states)} hidden states + logits {tuple(trace.logits.shape)} to {a.out}.safetensors")
    return 0


def _cmd_trace(a) -> int:
    from .engine import Engine

    eng = Engine.load(a.checkpoint, max_seq_len=max(a.seq_len, 64), max_seqs=a.batch,
                      mode_b_cache_bytes=a.mode_b_cache_mib << 20)
    ids = _input_ids(a, eng.cfg.vocab_size)
    if a.text:
        ids = torch.tensor([eng.tokenizer.encode(a.text).ids] * a.batch)
    trace = eng.trace(ids, decode_from=a.decode_from)
    trace.save(a.out)
    print(f"saved {len(trace.hidden_states)} hidden states + logits {tuple(trace.logits.shape)} to {a.out}.safetensors")
    return 0


def _cmd_generate(a) -> int:
    import time

    from .engine import Engine

    eng = Engine.load(a.checkpoint, max_seq_len=a.max_seq_len, mode_b_cache_bytes=a.mode_b_cache_mib << 20)
    if eng.tokenizer is None:
        raise SystemExit("no tokenizer.json next to the checkpoint")
    prompts = [eng.tokenizer.encode(p).ids for p in a.prompt]
    t0 = time.perf_counter()
    res = eng.generate(prompts, max_new_tokens=a.max_new_tokens, temperature=a.temperature, seed=a.seed)
    dt = time.perf_counter() - t0
    for p, toks in zip(a.prompt, res.tokens):
        print(f"--- {p!r}\n{eng.tokenizer.decode(toks)}")
    n = sum(len(t) for t in res.tokens)
    print(f"[{n} tokens in {dt:.2f}s]", file=sys.stderr)
    return 0


def _cmd_bench(a) -> int:
    """Prefill and decode throughput of the engine, with peak resident memory."""
    import os
    import resource
    import time

    from .engine import Engine
    from .engine.standard import StandardEngine

    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    t0 = time.perf_counter()
    eng = Engine.load(a.checkpoint, max_seq_len=a.prompt_len + a.gen_len + 1, max_seqs=a.batch,
                      max_tokens=a.batch * (a.prompt_len + a.gen_len + 64), mode_b_cache_bytes=a.mode_b_cache_mib << 20,
                      expert_cache_bytes=None if a.expert_cache_mib is None else a.expert_cache_mib << 20)
    t_load = time.perf_counter() - t0
    prompts = torch.randint(0, eng.cfg.vocab_size, (a.batch, a.prompt_len))
    if isinstance(eng, StandardEngine):
        t0 = time.perf_counter()
        logits, _, cache = eng.model.forward(prompts)
        t_pre = time.perf_counter() - t0
        nxt = logits[:, -1].argmax(-1)
        t0 = time.perf_counter()
        for _ in range(a.gen_len):
            logits, _, cache = eng.model.forward(nxt[:, None], cache)
            nxt = logits[:, -1].argmax(-1)
        t_dec = time.perf_counter() - t0
        pages, store, backend = None, eng.model.store, "torch_standard"
    else:
        cache = eng.new_cache()
        seqs = [cache.new_sequence() for _ in range(a.batch)]
        t0 = time.perf_counter()
        logits = eng.model.forward(list(zip(seqs, prompts.tolist())), cache)
        t_pre = time.perf_counter() - t0
        nxt = logits.argmax(-1).tolist()
        t0 = time.perf_counter()
        for _ in range(a.gen_len):
            nxt = eng.model.forward([(q, [t]) for q, t in zip(seqs, nxt)], cache).argmax(-1).tolist()
        t_dec = time.perf_counter() - t0
        pages, store, backend = cache.pages_in_use, eng.model.w, "torch_ref"
    peak_rss_bytes = _rss_bytes(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, sys.platform)
    res = {
        "checkpoint": a.checkpoint,
        "batch": a.batch,
        "prompt_len": a.prompt_len,
        "gen_len": a.gen_len,
        "threads": torch.get_num_threads(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_dynamic": os.environ.get("MKL_DYNAMIC"),
        "load_s": round(t_load, 3),
        "prefill_tok_s": round(a.batch * a.prompt_len / t_pre, 2),
        "decode_tok_s": round(a.batch * a.gen_len / t_dec, 3) if a.gen_len else None,
        "peak_rss_bytes": peak_rss_bytes,
        "peak_rss_gib": round(peak_rss_bytes / (1 << 30), 4),
        "kv_pages_in_use": pages,
        "kernel_backend": backend,
        "mode_b_cache_budget_bytes": store.mode_b_cache_bytes,
        "mode_b_cache_used_bytes": store.mode_b_cache_used,
        "mode_b_cache_hits": store.mode_b_cache_hits,
        "mode_b_cache_misses": store.mode_b_cache_misses,
        "mode_b_cache_evictions": store.mode_b_cache_evictions,
        "expert_cache": eng.model.expert_cache.stats(),
    }
    print(json.dumps(res, indent=2))
    return 0


def _cmd_compare(a) -> int:
    from .reference import ReferenceTrace, compare_traces

    cmp = compare_traces(ReferenceTrace.load(a.reference), ReferenceTrace.load(a.candidate))
    print(cmp.table())
    return 0 if cmp.min_cos >= a.min_cos else 1


def _cmd_export(a) -> int:
    from .format.export import export
    from .format.policy import Policy

    pol = Policy.load(a.policy)
    hessians = None
    if any(r.gptq for r in pol.rules):
        from .engine import Engine
        from .format.calibrate import calibrate

        if not a.calib_text:
            raise SystemExit("the policy asks for GPTQ: pass --calib-text (a UTF-8 text file)")
        eng = Engine.load(a.source, max_seq_len=a.calib_len + 8)
        if eng.tokenizer is None:
            raise SystemExit("no tokenizer.json next to the checkpoint")
        toks = eng.tokenizer.encode(open(a.calib_text, encoding="utf-8").read()).ids
        seqs = [toks[i : i + a.calib_len] for i in range(0, len(toks) - a.calib_len + 1, a.calib_len)][: a.calib_seqs]
        if not seqs:
            raise SystemExit(f"calibration text is shorter than --calib-len {a.calib_len} tokens")
        layers = {int(x) for x in a.calib_layers.split(",")} if a.calib_layers else None
        hessians = calibrate(eng, seqs, pol, layers, min_tokens=a.calib_min_tokens)
    out = export(a.source, a.out_dir, pol, hessians=hessians, shard_bytes=int(a.shard_gb * 2**30))
    s = json.loads((out / "manifest.json").read_text())["summary"]
    print(json.dumps(s, indent=2))
    return 0


def _cmd_gguf(a) -> int:
    from .format.gguf_export import merge_into_base, write_ternary_only

    if a.base:
        info = merge_into_base(a.base, a.export_dir, a.out, a.packing)
    else:
        info = write_ternary_only(a.export_dir, a.out, a.packing)
    print(json.dumps({"out": str(a.out), "bytes": Path(a.out).stat().st_size, "qtype": info["qtype"],
                      "converted": len(info["converted"]), "skipped": info["skipped"]}, indent=2))
    return 0


def _cmd_cast(a) -> int:
    from .checkpoint.cast import cast_checkpoint

    cast_checkpoint(a.source, a.out, a.dtype)
    return 0


def _cmd_qat_tokenize(a) -> int:
    from .qat.data import tokenize_text_file

    out = tokenize_text_file(a.text, a.tokenizer, a.out)
    print(out.with_suffix(".json").read_text())
    return 0


def _cmd_qat_teacher(a) -> int:
    from transformers import AutoModelForCausalLM

    from .qat.data import TokenWindows
    from .qat.teacher import precompute

    dev = a.device if a.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(a.teacher, dtype=torch.bfloat16 if dev == "cuda" else torch.float32)
    windows = TokenWindows(a.tokens, a.seq_len)
    n = min(len(windows), a.windows) if a.windows else len(windows)
    precompute(model, windows, a.out, list(range(n)), k=a.k, temperature=a.temperature, batch=a.batch, device=dev)
    return 0


def _cmd_qat_train(a) -> int:
    from .qat.train import main as train_main

    return train_main([str(a.config)] + (["--max-steps", str(a.max_steps)] if a.max_steps is not None else [])
                      + (["--plan"] if a.plan else []))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="boilerplate")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("inspect", help="derive the architecture spec of a checkpoint (local dir or HF repo id)")
    s.add_argument("checkpoint")
    s.add_argument("--revision")
    s.add_argument("--no-headers", action="store_true", help="skip shard headers (names + config only)")
    s.add_argument("--json", help="write the full spec as JSON")
    s.set_defaults(fn=_cmd_inspect)

    s = sub.add_parser("make-tiny", help="write a tiny random-init V4.1-format checkpoint")
    s.add_argument("out_dir")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=_cmd_make_tiny)

    s = sub.add_parser("make-truncated", help="cut a small trained model (layer + expert subset) out of a real checkpoint")
    s.add_argument("source", help="HF repo id or local checkpoint directory")
    s.add_argument("out_dir")
    s.add_argument("--layers", default="0,1,2,3,20,21,24", help="comma-separated source layer ids, in order")
    s.add_argument("--experts", type=int, default=16, help="keep the first N routed experts per layer")
    s.add_argument("--keep-engram", action="store_true")
    s.add_argument("--revision", default="main")
    s.set_defaults(fn=_cmd_make_truncated)

    s = sub.add_parser("reference", help="run a trusted reference and save a trace")
    s.add_argument("backend", choices=["official", "hf"])
    s.add_argument("checkpoint")
    s.add_argument("--out", required=True, help="trace path prefix")
    s.add_argument("--code-dir", help="official: the checkpoint's inference/ directory")
    s.add_argument("--work-dir", help="official: where to write the converted checkpoint")
    s.add_argument("--gpu-kernels", action="store_true", help="official: use the real TileLang kernels")
    s.add_argument("--text-only", action="store_true", help="official: drop the vision tower")
    s.add_argument("--tokenizer", help="official: directory with tokenizer.json (default: the checkpoint); needed for engram")
    s.add_argument("--seq-len", type=int, default=32)
    s.add_argument("--batch", type=int, default=1)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--decode-from", type=int, help="official: prefill this many tokens, then decode the rest one by one")
    s.add_argument("--text", help="trace this text (tokenized with the checkpoint's tokenizer) instead of random ids")
    s.set_defaults(fn=_cmd_reference)

    s = sub.add_parser("trace", help="run the Boilerplate engine (Mode A) and save a trace comparable to a reference trace")
    s.add_argument("checkpoint")
    s.add_argument("--out", required=True, help="trace path prefix")
    s.add_argument("--seq-len", type=int, default=32)
    s.add_argument("--batch", type=int, default=1)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--text", help="trace this text instead of random token ids")
    s.add_argument("--decode-from", type=int, help="prefill this many tokens, then decode the rest one by one")
    s.add_argument("--mode-b-cache-mib", type=int, default=256, help="maximum decoded Mode B bytes retained in the LRU (MiB)")
    s.set_defaults(fn=_cmd_trace)

    s = sub.add_parser("generate", help="generate text with the Boilerplate engine")
    s.add_argument("checkpoint")
    s.add_argument("prompt", nargs="+")
    s.add_argument("--max-new-tokens", type=int, default=32)
    s.add_argument("--temperature", type=float, default=0.0)
    s.add_argument("--seed", type=int)
    s.add_argument("--max-seq-len", type=int, default=4096)
    s.add_argument("--mode-b-cache-mib", type=int, default=256, help="maximum decoded Mode B bytes retained in the LRU (MiB)")
    s.set_defaults(fn=_cmd_generate)

    s = sub.add_parser("bench", help="measure engine prefill/decode throughput and peak memory")
    s.add_argument("checkpoint")
    s.add_argument("--prompt-len", type=int, default=128)
    s.add_argument("--gen-len", type=int, default=16)
    s.add_argument("--batch", type=int, default=1)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--threads", type=int, default=4, help="PyTorch CPU math threads")
    s.add_argument("--mode-b-cache-mib", type=int, default=256, help="maximum decoded Mode B bytes retained in the LRU (MiB)")
    s.add_argument("--expert-cache-mib", type=int, help="bound on materialized routed experts (default: keep all)")
    s.set_defaults(fn=_cmd_bench)

    s = sub.add_parser("serve", help="OpenAI-compatible HTTP server (/v1/completions, /v1/chat/completions, streaming)")
    s.add_argument("checkpoint")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--name", help="model id reported by /v1/models (default: the directory name)")
    s.add_argument("--max-seq-len", type=int, default=4096)
    s.add_argument("--default-max-tokens", type=int, default=256)
    s.add_argument("--threads", type=int, default=4, help="PyTorch CPU math threads")
    s.add_argument("--mode-b-cache-mib", type=int, default=256, help="maximum decoded Mode B bytes retained in the LRU (MiB)")
    s.add_argument("--expert-cache-mib", type=int, help="bound on materialized routed experts (default: keep all)")
    s.set_defaults(fn=lambda a: __import__("boilerplate.serve", fromlist=["main"]).main(a))

    s = sub.add_parser("lm-eval", help="run lm-evaluation-harness tasks in process on a checkpoint (needs lm_eval)")
    s.add_argument("checkpoint")
    s.add_argument("--tasks", required=True, help="comma-separated harness task names")
    s.add_argument("--limit", type=float, help="examples per task (or a fraction)")
    s.add_argument("--num-fewshot", type=int)
    s.add_argument("--seed", type=int, default=1234)
    s.add_argument("--max-seq-len", type=int, default=4096)
    s.add_argument("--threads", type=int, default=4, help="PyTorch CPU math threads")
    s.add_argument("--mode-b-cache-mib", type=int, default=256)
    s.add_argument("--out", help="write the results JSON here")
    s.set_defaults(fn=lambda a: __import__("boilerplate.evals.lm_eval_adapter", fromlist=["main"]).main(a))

    s = sub.add_parser("compare", help="compare two traces layer by layer")
    s.add_argument("reference")
    s.add_argument("candidate")
    s.add_argument("--min-cos", type=float, default=0.99)
    s.set_defaults(fn=_cmd_compare)

    s = sub.add_parser("export", help="write a Boilerplate-format checkpoint under a precision policy (Modes A/B/C)")
    s.add_argument("source")
    s.add_argument("out_dir")
    s.add_argument("--policy", required=True, help="YAML policy (see configs/policy/)")
    s.add_argument("--shard-gb", type=float, default=2.0)
    s.add_argument("--calib-text", help="GPTQ: text file to calibrate on")
    s.add_argument("--calib-seqs", type=int, default=512, help="GPTQ: number of calibration sequences")
    s.add_argument("--calib-len", type=int, default=512, help="GPTQ: tokens per calibration sequence")
    s.add_argument("--calib-layers", help="GPTQ: only calibrate these layer ids (comma-separated)")
    s.add_argument("--calib-min-tokens", type=int, default=1,
                   help="GPTQ: weights that saw fewer calibration tokens use the plain fit (recorded in the manifest)")
    s.set_defaults(fn=_cmd_export)

    s = sub.add_parser("gguf", help="write single-term ternary weights (k=1, group 256, fp16) as llama.cpp TQ1_0/TQ2_0")
    s.add_argument("export_dir", help="a Boilerplate export of a llama/olmoe checkpoint")
    s.add_argument("out", help="output .gguf")
    s.add_argument("--base", help="llama.cpp BF16/F32 conversion of the same source; its other tensors and metadata are kept")
    s.add_argument("--packing", choices=("tq1", "tq2"), default="tq1")
    s.set_defaults(fn=_cmd_gguf)

    s = sub.add_parser("cast", help="cast a safetensors checkpoint to another float dtype (e.g. OLMoE F32 -> BF16)")
    s.add_argument("source")
    s.add_argument("out")
    s.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    s.set_defaults(fn=_cmd_cast)

    s = sub.add_parser("qat-tokenize", help="tokenize a text file into a QAT token file (.bin + .json)")
    s.add_argument("text")
    s.add_argument("tokenizer", help="tokenizer.json")
    s.add_argument("out", help="output .bin")
    s.set_defaults(fn=_cmd_qat_tokenize)

    s = sub.add_parser("qat-teacher", help="precompute teacher top-k log-probs per training window (offline distillation)")
    s.add_argument("teacher", help="HF checkpoint of the unquantized teacher")
    s.add_argument("tokens", help="token file from qat-tokenize")
    s.add_argument("out")
    s.add_argument("--seq-len", type=int, default=2048)
    s.add_argument("--k", type=int, default=32)
    s.add_argument("--temperature", type=float, default=1.0)
    s.add_argument("--batch", type=int, default=1)
    s.add_argument("--windows", type=int, help="only the first N windows")
    s.add_argument("--device", default="auto")
    s.set_defaults(fn=_cmd_qat_teacher)

    s = sub.add_parser("qat-train", help="ternary QAT from a YAML run config (runs over 20 steps need approval: in the config)")
    s.add_argument("config")
    s.add_argument("--max-steps", type=int)
    s.add_argument("--plan", action="store_true", help="print steps/tokens and exit")
    s.set_defaults(fn=_cmd_qat_train)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())

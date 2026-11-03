"""Command-line entry point: ``boilerplate <command>``."""

from __future__ import annotations

import argparse
import json
import sys

import torch


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


def _input_ids(a, vocab: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(a.seed)
    return torch.randint(0, vocab, (a.batch, a.seq_len), generator=g)


def _cmd_reference(a) -> int:
    if a.backend == "official":
        from .reference.official_backend import OfficialReference

        ref = OfficialReference(
            a.code_dir, a.checkpoint, a.work_dir or f"{a.out}.converted", use_cpu_kernels=not a.gpu_kernels,
            text_only=a.text_only, max_seq_len=max(a.seq_len, 64), max_batch_size=a.batch,
        )
        vocab = ref.args_kwargs["vocab_size"]
    else:
        from .reference.hf_backend import HFReference

        ref = HFReference(a.checkpoint)
        vocab = ref.model.config.vocab_size
    trace = ref.trace(_input_ids(a, vocab))
    trace.save(a.out)
    print(f"saved {len(trace.hidden_states)} hidden states + logits {tuple(trace.logits.shape)} to {a.out}.safetensors")
    return 0


def _cmd_compare(a) -> int:
    from .reference import ReferenceTrace, compare_traces

    cmp = compare_traces(ReferenceTrace.load(a.reference), ReferenceTrace.load(a.candidate))
    print(cmp.table())
    return 0 if cmp.min_cos >= a.min_cos else 1


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

    s = sub.add_parser("reference", help="run a trusted reference and save a trace")
    s.add_argument("backend", choices=["official", "hf"])
    s.add_argument("checkpoint")
    s.add_argument("--out", required=True, help="trace path prefix")
    s.add_argument("--code-dir", help="official: the checkpoint's inference/ directory")
    s.add_argument("--work-dir", help="official: where to write the converted checkpoint")
    s.add_argument("--gpu-kernels", action="store_true", help="official: use the real TileLang kernels")
    s.add_argument("--text-only", action="store_true", help="official: drop the vision tower")
    s.add_argument("--seq-len", type=int, default=32)
    s.add_argument("--batch", type=int, default=1)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=_cmd_reference)

    s = sub.add_parser("compare", help="compare two traces layer by layer")
    s.add_argument("reference")
    s.add_argument("candidate")
    s.add_argument("--min-cos", type=float, default=0.99)
    s.set_defaults(fn=_cmd_compare)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())

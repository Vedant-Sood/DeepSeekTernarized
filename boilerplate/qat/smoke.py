"""Distributed QAT dry run on a tiny random Hugging Face model (no pretrained weights, no GPU).

``python -m boilerplate.qat.smoke --world 2 --model olmoe`` spawns ``world`` CPU processes on the
``gloo`` backend and runs the real training path end to end at toy size: convert block linears to
BitLinear, freeze routers, FSDP2-shard, AdamW with the lambda warmup, log routing drift against the
unquantized start, save a sharded checkpoint, resume it in fresh processes and check that the
resumed step reproduces the uninterrupted one exactly. It is what the tests run, and the dry run
the M4 plan asks for before any real (approval-gated) QAT launch.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def tiny_model(kind: str, seed: int = 5):
    import transformers

    torch.manual_seed(seed)
    if kind == "llama":
        config = transformers.LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                          num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)
        return transformers.LlamaForCausalLM(config).float()
    config = transformers.OlmoeConfig(vocab_size=64, hidden_size=32, intermediate_size=32, num_hidden_layers=2,
                                      num_attention_heads=4, num_key_value_heads=2, num_experts=4,
                                      num_experts_per_tok=2, max_position_embeddings=64)
    return transformers.OlmoeForCausalLM(config).float()


def unfuse_experts(model) -> None:
    """Recent transformers stores OLMoE experts as fused 3-D parameters. Rebuild them as one
    ``nn.Linear`` per expert projection (the published checkpoint layout), so BitLinear can wrap them."""
    from torch import nn

    for layer in getattr(model.model, "layers", []):
        experts = getattr(layer.mlp, "experts", None)
        if experts is None or not hasattr(experts, "gate_up_proj"):
            continue
        gate_up, down = experts.gate_up_proj.data, experts.down_proj.data  # [E, 2I, H], [E, H, I]
        n = gate_up.size(0)
        inter = gate_up.size(1) // 2

        class Expert(nn.Module):
            def __init__(self, gu, d):
                super().__init__()
                g, u = gu.chunk(2, 0)
                self.gate_proj = nn.Linear(g.size(1), inter, bias=False)
                self.up_proj = nn.Linear(u.size(1), inter, bias=False)
                self.down_proj = nn.Linear(inter, d.size(0), bias=False)
                with torch.no_grad():
                    self.gate_proj.weight.copy_(g)
                    self.up_proj.weight.copy_(u)
                    self.down_proj.weight.copy_(d)

            def forward(self, x):
                return self.down_proj(nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))

        class Experts(nn.ModuleList):
            def forward(self, hidden, top_k_index, top_k_weights):
                out = torch.zeros_like(hidden)
                for e, expert in enumerate(self):
                    token, slot = torch.where(top_k_index == e)
                    if token.numel():
                        out.index_add_(0, token, expert(hidden[token]) * top_k_weights[token, slot, None].to(hidden.dtype))
                return out

        layer.mlp.experts = Experts([Expert(gate_up[e], down[e]) for e in range(n)])


def _batch(rank: int, step: int, batch: int, seq: int, vocab: int) -> torch.Tensor:
    return torch.randint(0, vocab, (batch, seq), generator=torch.Generator().manual_seed(1000 * step + rank))


def _build(kind: str, warmup: int):
    from .distributed import convert_linears, shard
    from .routing import freeze_routers

    model = tiny_model(kind)
    if kind == "olmoe":
        unfuse_experts(model)
    converted = convert_linears(model, group=16, master_dtype=torch.float32)
    frozen = freeze_routers(model) if kind == "olmoe" else 0
    shard(model)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    return model, optimizer, converted, frozen


def _step(model, optimizer, rank, step, warmup, kind, routing_log):
    from .bitlinear import warmup_level
    from .distributed import set_quantization_level
    from .routing import capture_hf_routing

    set_quantization_level(model, warmup_level(step, warmup))
    ids = _batch(rank, step, 2, 12, model.config.vocab_size)
    if kind == "olmoe":
        with capture_hf_routing(model, model.config.num_experts_per_tok) as calls:
            loss = model(ids, labels=ids).loss
        routing_log.append(calls[0])
    else:
        loss = model(ids, labels=ids).loss
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def _worker(rank: int, world: int, port: int, kind: str, steps: int, warmup: int, out: str, resume: bool) -> None:
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        from .distributed import load_sharded, save_sharded
        from .routing import RoutingDrift

        model, optimizer, converted, frozen = _build(kind, warmup)
        ckpt = Path(out) / "ckpt"
        losses, routing = [], []
        if not resume:
            for step in range(steps):
                losses.append(_step(model, optimizer, rank, step, warmup, kind, routing))
                if step == steps - 2:
                    save_sharded(ckpt, model, optimizer, step=step + 1, tokens_seen=(step + 1) * 24 * world)
            start = 0
        else:
            progress = load_sharded(ckpt, model, optimizer)
            start = progress["step"]
            for step in range(start, steps):
                losses.append(_step(model, optimizer, rank, step, warmup, kind, routing))
        report = {"rank": rank, "start": start, "losses": losses, "converted": len(converted),
                  "frozen_routers": frozen}
        if kind == "olmoe" and not resume:
            # Routing drift on the same tokens: step 0 ran the unquantized start (level 0); rerun
            # that batch through the trained, fully quantized model and compare expert choices.
            from .routing import capture_hf_routing

            with torch.no_grad(), capture_hf_routing(model, model.config.num_experts_per_tok) as calls:
                model(_batch(rank, 0, 2, 12, model.config.vocab_size))
            drift = RoutingDrift(len(routing[0]), model.config.num_experts)
            drift.update(routing[0], calls[0])
            report["routing"] = drift.summary()
        Path(out, f"{'resume' if resume else 'run'}{rank}.json").write_text(json.dumps(report))
    finally:
        dist.destroy_process_group()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def run(world: int = 2, kind: str = "olmoe", steps: int = 3, warmup: int = 2, out: str | None = None) -> dict:
    """Uninterrupted run, then a resume from the checkpoint before the last step, in fresh processes."""
    if steps < 2:
        raise ValueError("need at least two steps to checkpoint before the last one")
    out = out or tempfile.mkdtemp(prefix="qat-smoke-")
    for resume in (False, True):
        mp.spawn(_worker, args=(world, _free_port(), kind, steps, warmup, out, resume), nprocs=world, join=True)
    runs = [json.loads(Path(out, f"run{r}.json").read_text()) for r in range(world)]
    resumes = [json.loads(Path(out, f"resume{r}.json").read_text()) for r in range(world)]
    return {"out": out, "runs": runs, "resumes": resumes,
            "resume_exact": all(r["losses"][-1] == s["losses"][-1] for r, s in zip(runs, resumes))}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--world", type=int, default=2)
    p.add_argument("--model", choices=("llama", "olmoe"), default="olmoe")
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--warmup", type=int, default=2)
    a = p.parse_args(argv)
    result = run(a.world, a.model, a.steps, a.warmup)
    print(json.dumps(result, indent=2))
    return 0 if result["resume_exact"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

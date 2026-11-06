"""Ternary QAT training loop (BitNet b1.58 style) driven by one YAML run config.

    boilerplate qat-train run.yaml                       # one process (CPU or one GPU)
    torchrun --nproc-per-node 4 -m boilerplate.qat.train run.yaml   # FSDP2 across GPUs

What one run does: load the BF16 source, swap block linears for BitLinear, freeze routers,
optionally FSDP-shard, then train on fixed-length windows of a token file with AdamW, linear LR
warmup + cosine decay, the lambda warmup (full precision → fully ternary over
``quant_warmup_steps``), CE or CE + KL to stored teacher top-k, gradient clipping. It logs JSON
lines (loss, LR, lambda, tokens, throughput, routing drift on a fixed probe batch), checkpoints
every ``checkpoint_every`` steps and at the end, resumes exactly from the latest checkpoint, and
exports Mode D at the end.

**Approval gate.** A run longer than ``SMOKE_STEPS`` optimizer steps refuses to start unless the
config carries ``approval:`` (who approved it and when). The project rule is that small-model QAT
needs an approved GPU budget and full-size QAT needs Vedant's explicit confirmation; this makes
an accidental launch fail fast instead of burning a budget.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import yaml

SMOKE_STEPS = 20


@dataclass
class RunConfig:
    source: str                      # HF checkpoint directory (BF16 or F32)
    train_tokens: str                # token file from ``qat-tokenize``
    out_dir: str
    token_budget: int                # tokens to train on (all ranks together)
    seq_len: int = 2048
    micro_batch: int = 1             # windows per rank per micro step
    grad_accum: int = 1
    lr: float = 1.5e-4
    min_lr: float = 1.5e-5
    weight_decay: float = 0.1
    lr_warmup_steps: int = 100
    quant_warmup_steps: int = 500    # lambda: 0 → 1 over these steps
    group: int = 128
    scale_dtype: str = "fp16"
    master_dtype: str = "bf16"
    packing: str = "tq1"             # Mode D export packing
    freeze_router: bool = True
    grad_clip: float = 1.0
    optimizer: str = "adamw"         # "adamw" or "adamw8bit" (bitsandbytes, for 8 GB GPUs)
    gradient_checkpointing: bool = False
    teacher_dir: str | None = None   # offline teacher top-k (``qat-teacher``); None = CE only
    kl_weight: float = 0.5
    seed: int = 0
    log_every: int = 10
    checkpoint_every: int = 500
    routing_every: int = 100         # MoE only
    probe_windows: int = 4
    max_steps: int | None = None     # stop early (smoke tests); does not change the schedule
    export_at_tokens: list = field(default_factory=list)  # extra Mode D snapshots (e.g. [1e9, 3e9])
    device: str = "auto"
    approval: str | None = None
    extra: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "RunConfig":
        return cls(**yaml.safe_load(Path(path).read_text()))


DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def lr_at(step: int, total: int, cfg: RunConfig) -> float:
    if step < cfg.lr_warmup_steps:
        return cfg.lr * (step + 1) / cfg.lr_warmup_steps
    progress = (step - cfg.lr_warmup_steps) / max(1, total - cfg.lr_warmup_steps)
    return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1 + math.cos(math.pi * min(progress, 1.0)))


def plan(cfg: RunConfig, world: int) -> dict:
    tokens_per_step = cfg.seq_len * cfg.micro_batch * cfg.grad_accum * world
    total = max(1, cfg.token_budget // tokens_per_step)
    return {"tokens_per_step": tokens_per_step, "total_steps": total, "world": world}


def _optimizer(params, cfg: RunConfig):
    if cfg.optimizer == "adamw8bit":
        import bitsandbytes as bnb  # optional dependency for small GPUs

        return bnb.optim.AdamW8bit(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95))


def train(cfg: RunConfig, log=print) -> dict:
    import torch.distributed as dist

    from .bitlinear import warmup_level
    from .data import Sampler, TokenWindows, batch
    from .distill import distillation_loss
    from .distributed import load_sharded, save_sharded, set_quantization_level, shard
    from .model import load_for_qat
    from .routing import RoutingDrift, capture_hf_routing

    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    p = plan(cfg, world)
    if (cfg.max_steps or p["total_steps"]) > SMOKE_STEPS and not cfg.approval:
        raise SystemExit(f"this run is {p['total_steps']} steps ({cfg.token_budget:,} tokens); runs over "
                         f"{SMOKE_STEPS} steps need `approval:` in the config (approved GPU budget)")
    device = cfg.device if cfg.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    if world > 1:
        dist.init_process_group("nccl" if device == "cuda" else "gloo")
        if device == "cuda":
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    torch.manual_seed(cfg.seed + rank)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model, info = load_for_qat(cfg.source, dtype=DTYPES[cfg.master_dtype], group=cfg.group,
                               scale_dtype=DTYPES[cfg.scale_dtype], freeze_router=cfg.freeze_router)
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.to(device)
    if world > 1:
        shard(model)
    params = [q for q in model.parameters() if q.requires_grad]
    optimizer = _optimizer(params, cfg)
    windows = TokenWindows(cfg.train_tokens, cfg.seq_len)
    teacher, pool = None, None
    if cfg.teacher_dir:
        from .teacher import TeacherStore

        teacher = TeacherStore(cfg.teacher_dir)
        tm = teacher.meta
        if tm["seq_len"] != cfg.seq_len or tm.get("token_file_sha256") != windows.meta.get("sha256"):
            raise SystemExit(f"teacher {cfg.teacher_dir} was built for seq_len {tm['seq_len']} on token file "
                             f"{tm.get('token_file_sha256')}; this run uses seq_len {cfg.seq_len} on "
                             f"{windows.meta.get('sha256')}")
        if len(teacher.where) < len(windows):  # train only on windows that have teacher outputs
            pool = sorted(teacher.where)
            if rank == 0:
                log(json.dumps({"event": "teacher_subset", "windows": len(pool), "of": len(windows)}))
    sampler = Sampler(len(pool) if pool else len(windows), seed=cfg.seed, rank=rank, world=world)
    moe = info["model_type"] == "olmoe"
    probe_ids = list(range(cfg.probe_windows))
    probe = batch(windows, probe_ids)[0].to(device)
    initial_routing = None

    ckpt_single, ckpt_dist = out / "checkpoint.pt", out / "checkpoint"
    start, tokens_seen = 0, 0
    if world == 1 and ckpt_single.exists():
        from .checkpoint import load_checkpoint

        progress = load_checkpoint(ckpt_single, model, optimizer)
    elif world > 1 and (ckpt_dist / "state").exists():
        progress = load_sharded(ckpt_dist, model, optimizer)
    else:
        progress = None
    if progress:
        start, tokens_seen = progress["step"], progress["tokens_seen"]
        sampler.load_state(progress["trainer_state"]["sampler"])
        initial_routing = progress["trainer_state"].get("initial_routing")
    if moe and initial_routing is None:  # routing of the unquantized model on the probe batch
        set_quantization_level(model, 0.0)
        with torch.no_grad(), capture_hf_routing(model, model.config.num_experts_per_tok) as calls:
            model(probe)
        initial_routing = [r.tolist() for r in calls[0]]

    def save(step):
        state = {"sampler": sampler.state(), "initial_routing": initial_routing, "config": asdict(cfg)}
        if world == 1:
            from .checkpoint import save_checkpoint

            save_checkpoint(ckpt_single, model, optimizer, step=step, tokens_seen=tokens_seen, trainer_state=state)
        else:
            save_sharded(ckpt_dist, model, optimizer, step=step, tokens_seen=tokens_seen, trainer_state=state)

    def export_now(path):
        """Mode D export of the current weights (every rank gathers, rank 0 writes)."""
        from .export import export_mode_d

        state = None
        if world > 1:
            from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

            state = get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
        if rank != 0:
            return None
        # the export ternarizes the master weights itself, so it does not depend on lambda
        return export_mode_d(model, cfg.source, path, packing=cfg.packing, state_dict=state, log=log)

    log_file = (out / f"log.rank{rank}.jsonl").open("a")
    end = min(p["total_steps"], cfg.max_steps) if cfg.max_steps else p["total_steps"]
    if rank == 0:
        log(json.dumps({"event": "start", **p, "start_step": start, "end_step": end, "device": device,
                        "converted": len(info["converted"]), "frozen_routers": info["frozen_routers"]}))
    model.train()
    t_last = time.perf_counter()
    for step in range(start, end):
        level = warmup_level(step, cfg.quant_warmup_steps)
        set_quantization_level(model, level)
        lr = lr_at(step, p["total_steps"], cfg)
        for g in optimizer.param_groups:
            g["lr"] = lr
        total_loss = 0.0
        for _ in range(cfg.grad_accum):
            ids = sampler.next(cfg.micro_batch)
            if pool:
                ids = [pool[i] for i in ids]
            x, y = batch(windows, ids)
            x, y = x.to(device), y.to(device)
            logits = model(x).logits
            if teacher is not None:
                loss = distillation_loss(logits, y, teacher.get(ids), kl_weight=cfg.kl_weight)
            else:
                loss = torch.nn.functional.cross_entropy(logits.float().flatten(0, 1), y.flatten())
            (loss / cfg.grad_accum).backward()
            total_loss += float(loss.detach()) / cfg.grad_accum
        if cfg.grad_clip:
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        tokens_seen += p["tokens_per_step"]
        record = {"step": step + 1, "loss": total_loss, "lr": lr, "lambda": level, "tokens": tokens_seen}
        if (step + 1) % cfg.log_every == 0 or step + 1 == end:
            now = time.perf_counter()
            record["tok_s"] = p["tokens_per_step"] * min(cfg.log_every, step + 1 - start) / (now - t_last)
            t_last = now
            if device == "cuda":
                record["max_vram_bytes"] = torch.cuda.max_memory_allocated()
        if moe and ((step + 1) % cfg.routing_every == 0 or step + 1 == end):
            with torch.no_grad(), capture_hf_routing(model, model.config.num_experts_per_tok) as calls:
                model(probe)
            drift = RoutingDrift(len(calls[0]), model.config.num_experts)
            drift.update([torch.tensor(r) for r in initial_routing], calls[0])
            record["routing"] = drift.summary()
        log_file.write(json.dumps(record) + "\n")
        log_file.flush()
        if rank == 0 and ("tok_s" in record):
            log(json.dumps(record))
        if (step + 1) % cfg.checkpoint_every == 0 or step + 1 == end:
            save(step + 1)
        for mark in cfg.export_at_tokens:  # quality-vs-tokens snapshots from one schedule
            if tokens_seen - p["tokens_per_step"] < float(mark) <= tokens_seen and step + 1 < p["total_steps"]:
                snap = export_now(out / f"mode_d_{cfg.packing}_{int(float(mark)):d}tok")
                if snap:
                    log(json.dumps({"event": "snapshot", "step": step + 1, "tokens": tokens_seen, "path": str(snap)}))
    log_file.close()

    result = {"steps": end, "tokens_seen": tokens_seen}
    if end == p["total_steps"]:  # finished the budget: export Mode D
        path = export_now(out / f"mode_d_{cfg.packing}")
        if path:
            result["mode_d"] = str(path)
    if world > 1:
        dist.destroy_process_group()
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("config", type=Path)
    ap.add_argument("--max-steps", type=int, help="override max_steps (smoke runs)")
    ap.add_argument("--plan", action="store_true", help="print the step plan and exit")
    a = ap.parse_args(argv)
    cfg = RunConfig.load(a.config)
    if a.max_steps is not None:
        cfg.max_steps = a.max_steps
    if a.plan:
        print(json.dumps(plan(cfg, int(os.environ.get("WORLD_SIZE", "1"))), indent=2))
        return 0
    print(json.dumps(train(cfg), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

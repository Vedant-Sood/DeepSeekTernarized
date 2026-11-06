"""Compute, memory and cost estimate for a QAT run config (``boilerplate qat-estimate run.yaml``).

Inputs: the run config (token budget, batch shape, dtypes, optimizer), the source checkpoint's
tensor shapes (read from safetensors headers; nothing is loaded) and the hardware. Outputs:

* parameters: total, routed-expert, and active per token (MoE: ``k / E`` of the expert weights);
* training FLOPs ``6 * active * tokens`` (forward + backward; BitLinear's quantize/dequantize is
  elementwise and ignored), plus the teacher's ``2 * active_teacher * tokens`` when distilling
  (inside the step for an online teacher, once beforehand for offline top-k, which also costs
  ``tokens * (4k + 4)`` bytes of disk: 132 B/token at k = 32);
* GPU memory per device: master weights, gradients and optimizer state (FSDP divides all three by
  the world size), plus an activation term per micro batch (rough; checkpointing shrinks it);
* time and cost: from a **measured** throughput (``tok_s`` from a probe run's log) when given,
  otherwise from peak dense BF16 FLOP/s times an assumed MFU. The assumption is printed with
  the result; the plan says to refine every estimate from a measured probe before asking for a
  budget.

The figures are planning numbers, not measurements; the report quotes them as such.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

# Dense BF16 tensor-core peak FLOP/s (vendor datasheets, no sparsity) and VRAM bytes.
GPUS = {
    "h100": {"flops": 989e12, "vram": 80e9},
    "a100": {"flops": 312e12, "vram": 80e9},
    "l40s": {"flops": 362e12, "vram": 48e9},
    "rtx4090": {"flops": 165e12, "vram": 24e9},
    "rtx4060-laptop": {"flops": 60e12, "vram": 8e9},  # approximate; boost clocks vary by laptop
}
EXPERT_RE = re.compile(r"\.experts\.\d+\.|\.experts\.(gate_up_proj|down_proj|w[123])")
BYTES = {"bf16": 2, "fp16": 2, "fp32": 4}


def param_counts(source: str | Path) -> dict:
    """Total, routed-expert and active-per-token parameter counts from safetensors headers."""
    from safetensors import safe_open

    source = Path(source)
    config = json.loads((source / "config.json").read_text())
    config = config.get("text_config", config)
    total = expert = 0
    for f in sorted(source.glob("*.safetensors")):
        with safe_open(str(f), "pt") as h:
            for name in h.keys():
                n = 1
                for d in h.get_slice(name).get_shape():
                    n *= d
                total += n
                expert += n if EXPERT_RE.search(name) else 0
    E = config.get("num_experts") or config.get("n_routed_experts") or 0
    k = config.get("num_experts_per_tok") or 0
    active = total - expert + (expert * k // E if E else 0)
    return {"total": total, "expert": expert, "active": active, "experts": E, "top_k": k,
            "hidden": config.get("hidden_size"), "layers": config.get("num_hidden_layers")}


def estimate(cfg, *, world: int = 1, gpu: str = "h100", mfu: float = 0.35, tok_s: float | None = None,
             price_per_gpu_hour: float | None = None, params: dict | None = None,
             teacher_params: int | None = None) -> dict:
    from .train import plan

    p = params or param_counts(cfg.source)
    hw = GPUS[gpu]
    steps = plan(cfg, world)
    tokens = steps["total_steps"] * steps["tokens_per_step"]
    train_flops = 6 * p["active"] * tokens
    teacher_active = teacher_params or p["active"]
    online = bool(getattr(cfg, "teacher_model", None))
    teacher_flops = 2 * teacher_active * tokens if (cfg.teacher_dir or online) else 0
    # offline top-k on disk: k int16/int32 ids + k fp16 log-probs + one fp32 tail per token
    teacher_disk = tokens * (getattr(cfg, "teacher_k", 32) * (2 + 2) + 4) if cfg.teacher_dir else 0
    w = BYTES[cfg.master_dtype]
    opt = 2 * (1 if cfg.optimizer == "adamw8bit" else 4)  # two moments per parameter
    state = p["total"] * (w + w + opt)  # weights + grads + optimizer
    hidden, layers = p.get("hidden") or 0, p.get("layers") or 0
    # ~34 bytes per token per hidden unit per layer for a BF16 transformer block without
    # checkpointing (Korthikanti et al. 2022, attention scores excluded); ~2 with checkpointing.
    per = 2 if cfg.gradient_checkpointing else 34
    activations = per * cfg.seq_len * cfg.micro_batch * hidden * layers
    teacher_weights = p["total"] * BYTES.get(getattr(cfg, "teacher_dtype", "bf16"), 2) if online else 0
    per_gpu = state / world + activations + teacher_weights  # the online teacher is replicated
    if tok_s:
        basis = f"measured {tok_s:,.0f} tok/s across {world} GPU(s)"
        seconds = tokens / tok_s
    else:
        basis = f"assumed MFU {mfu:.0%} of {gpu} peak {hw['flops'] / 1e12:.0f} TFLOP/s (BF16 dense)"
        # an online teacher's forward runs inside the training step
        seconds = (train_flops + (teacher_flops if online else 0)) / (world * hw["flops"] * mfu)
    gpu_hours = seconds * world / 3600
    out = {
        "params": p, "tokens": tokens, **steps,
        "train_flops": train_flops, "teacher_flops": teacher_flops,
        "teacher_mode": "online" if online else ("offline" if cfg.teacher_dir else "none"),
        "teacher_disk_bytes": int(teacher_disk), "teacher_weight_bytes_per_gpu": int(teacher_weights),
        "memory_per_gpu_bytes": int(per_gpu), "state_bytes_total": int(state), "activation_bytes": int(activations),
        "fits": per_gpu < 0.9 * hw["vram"], "gpu": gpu, "world": world,
        "wall_clock_hours": seconds / 3600, "gpu_hours": gpu_hours, "throughput_basis": basis,
    }
    if price_per_gpu_hour is not None:
        out["cost"] = gpu_hours * price_per_gpu_hour
        out["price_per_gpu_hour"] = price_per_gpu_hour
    return out


def main(a) -> int:
    from .train import RunConfig

    cfg = RunConfig.load(a.config)
    if a.token_budget:
        cfg.token_budget = int(a.token_budget)
    result = estimate(cfg, world=a.gpus, gpu=a.gpu, mfu=a.mfu, tok_s=a.tok_s, price_per_gpu_hour=a.price)
    print(json.dumps(result, indent=2))
    return 0

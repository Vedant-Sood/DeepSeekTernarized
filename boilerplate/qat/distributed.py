"""Model conversion and sharded (FSDP2) training state for ternary QAT.

* :func:`convert_linears` swaps selected ``nn.Linear`` modules for :class:`BitLinear`. The default
  pattern follows the M4 plan: every projection inside the decoder blocks (attention, dense MLP and
  MoE experts) becomes ternary; embeddings, LM head, norms and routers stay high precision.
* :func:`shard` applies PyTorch FSDP2 (``fully_shard``) per decoder block, then to the root, so
  parameters, gradients and optimizer state are split across ranks (ZeRO-3). The same code runs on
  ``gloo`` (CPU, used by the tests) and ``nccl`` (GPU).
* :func:`save_sharded` / :func:`load_sharded` write and restore the model, optimizer, progress and
  every rank's RNG state with ``torch.distributed.checkpoint``, so a resumed run continues exactly.

Expert parallelism: here MoE experts are sharded like every other parameter (each rank holds a
slice of every expert and all-gathers it for the forward). That is enough for OLMoE-1B-7B on
2-4 x 80 GB. True expert parallelism (each rank owns whole experts, tokens move by all-to-all) is
not implemented; the router capture in ``routing.py`` and the per-expert module layout are what it
would plug into.
"""

from __future__ import annotations

import random
import re
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import nn

from .bitlinear import BitLinear

DEFAULT_PATTERN = r"layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj|mlp\.experts\.\d+\.(gate|up|down)_proj)$"


def convert_linears(model: nn.Module, pattern: str = DEFAULT_PATTERN, *, group: int = 128,
                    scale_dtype: torch.dtype = torch.float16, master_dtype: torch.dtype = torch.bfloat16) -> list[str]:
    """Replace every ``nn.Linear`` whose qualified name matches ``pattern``; returns the names."""
    regex = re.compile(pattern)
    targets = [(name, m) for name, m in model.named_modules()
               if isinstance(m, nn.Linear) and not isinstance(m, BitLinear) and regex.search(name)]
    for name, linear in targets:
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child, BitLinear.from_linear(linear, group=group, scale_dtype=scale_dtype,
                                                     master_dtype=master_dtype))
    return [name for name, _ in targets]


def set_quantization_level(model: nn.Module, level: float) -> None:
    for m in model.modules():
        if isinstance(m, BitLinear):
            m.set_quantization_level(level)


def shard(model: nn.Module, block_pattern: str = r"layers\.\d+$", **fsdp_kwargs) -> nn.Module:
    """FSDP2-shard each module matching ``block_pattern`` (decoder blocks), then the root."""
    from torch.distributed.fsdp import fully_shard

    regex = re.compile(block_pattern)
    blocks = [m for name, m in model.named_modules() if name and regex.search(name)]
    for block in blocks:
        fully_shard(block, **fsdp_kwargs)
    fully_shard(model, **fsdp_kwargs)
    return model


def _rng_state(generator: torch.Generator | None) -> dict:
    # NumPy's state holds an ndarray, whose pickle needs arbitrary globals; store its keys as a
    # tensor so the rank file loads with torch.load(weights_only=True), as in qat/checkpoint.py.
    numpy_rng = np.random.get_state()
    numpy_rng = (numpy_rng[0], torch.from_numpy(numpy_rng[1].astype(np.int64)), *numpy_rng[2:])
    return {"python": random.getstate(), "numpy": numpy_rng, "torch": torch.get_rng_state(),
            "generator": generator.get_state() if generator is not None else None}


def save_sharded(path: str | Path, model: nn.Module, optimizer: torch.optim.Optimizer, *, step: int,
                 tokens_seen: int, trainer_state: dict | None = None,
                 generator: torch.Generator | None = None) -> Path:
    """Collective: every rank must call it. Writes a DCP directory plus one RNG file per rank."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict

    path = Path(path)
    model_state, optim_state = get_state_dict(model, optimizer)
    levels = {name: m.level for name, m in model.named_modules() if isinstance(m, BitLinear)}
    dcp.save({"model": model_state, "optimizer": optim_state}, checkpoint_id=str(path / "state"))
    rank = dist.get_rank() if dist.is_initialized() else 0
    torch.save({"version": 2, "step": step, "tokens_seen": tokens_seen, "trainer_state": trainer_state or {},
                "levels": levels, "rng": _rng_state(generator)}, path / f"rank{rank:05d}.pt")
    if dist.is_initialized():
        dist.barrier()
    return path


def load_sharded(path: str | Path, model: nn.Module, optimizer: torch.optim.Optimizer, *,
                 generator: torch.Generator | None = None) -> dict:
    """Collective inverse of :func:`save_sharded` on the same world size.

    The per-rank file is read with PyTorch's restricted weights-only unpickler, so it cannot run
    pickled code. Version 1 rank files (NumPy pickle globals) must be recreated; there is no
    unrestricted fallback. The DCP shards themselves are tensor data read by torch.distributed.
    """
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

    path = Path(path)
    model_state, optim_state = get_state_dict(model, optimizer)
    state = {"model": model_state, "optimizer": optim_state}
    dcp.load(state, checkpoint_id=str(path / "state"))
    set_state_dict(model, optimizer, model_state_dict=state["model"], optim_state_dict=state["optimizer"])
    rank = dist.get_rank() if dist.is_initialized() else 0
    local = torch.load(path / f"rank{rank:05d}.pt", map_location="cpu", weights_only=True)
    if not isinstance(local, dict) or local.get("version") != 2:
        raise ValueError("unsupported distributed checkpoint version (recreate version 1 checkpoints)")
    modules = dict(model.named_modules())
    unknown = [name for name in local["levels"] if not isinstance(modules.get(name), BitLinear)]
    if unknown:
        raise ValueError(f"checkpoint levels name modules that are not BitLinear: {unknown[:3]}")
    for name, level in local["levels"].items():
        modules[name].set_quantization_level(level)
    rng = local["rng"]
    if (rng["generator"] is None) != (generator is None):
        raise ValueError("checkpoint and caller must both supply the optional generator")
    random.setstate(rng["python"])
    numpy_rng = rng["numpy"]
    np.random.set_state((numpy_rng[0], numpy_rng[1].numpy().astype(np.uint32), *numpy_rng[2:]))
    torch.set_rng_state(rng["torch"])
    if generator is not None:
        generator.set_state(rng["generator"])
    return {"step": local["step"], "tokens_seen": local["tokens_seen"], "trainer_state": local["trainer_state"]}

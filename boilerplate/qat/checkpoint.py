"""Single-process QAT checkpoints, including optimizer and random-generator state.

The caller owns dataset/sampler position and can store it in ``trainer_state``. Distributed
checkpoint coordination and data-loader worker state are intentionally not implied here.
"""

from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch import nn


def save_checkpoint(path: str | Path, model: nn.Module, optimizer: torch.optim.Optimizer, *,
                    step: int, tokens_seen: int, trainer_state: dict | None = None,
                    generator: torch.Generator | None = None) -> Path:
    """Atomically save training state; trainer_state must use tensors and plain Python data."""
    if step < 0 or tokens_seen < 0:
        raise ValueError("step and tokens_seen must be nonnegative")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    numpy_rng = np.random.get_state()
    # NumPy's ndarray pickle requires arbitrary globals. Store the keys as a plain tensor
    # so the whole checkpoint can be read by PyTorch's restricted weights-only loader.
    numpy_rng = (numpy_rng[0], torch.from_numpy(numpy_rng[1].astype(np.int64)), *numpy_rng[2:])
    state = {
        "version": 2, "step": step, "tokens_seen": tokens_seen, "trainer_state": trainer_state or {},
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "module_training": {name: module.training for name, module in model.named_modules()},
        "requires_grad": {name: param.requires_grad for name, param in model.named_parameters()},
        "python_rng": random.getstate(), "numpy_rng": numpy_rng,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "generator_rng": generator.get_state() if generator is not None else None,
        "torch_threads": torch.get_num_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            torch.save(state, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
    return path


def load_checkpoint(path: str | Path, model: nn.Module, optimizer: torch.optim.Optimizer, *,
                    generator: torch.Generator | None = None) -> dict:
    """Restore tensor/plain-data state without executing checkpoint pickle globals.

    Version 1 checkpoints used NumPy pickle globals and must be recreated with this version;
    there is deliberately no fallback to unrestricted pickle deserialization.
    """
    state = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or state.get("version") != 2:
        raise ValueError("unsupported checkpoint version")
    if (state["generator_rng"] is None) != (generator is None):
        raise ValueError("checkpoint and caller must both supply the optional generator")
    cuda_rng = state["cuda_rng"]
    if cuda_rng is not None and (not torch.cuda.is_available() or len(cuda_rng) != torch.cuda.device_count()):
        raise ValueError("CUDA device count differs from the checkpoint")
    modules, params = dict(model.named_modules()), dict(model.named_parameters())
    if modules.keys() != state["module_training"].keys() or params.keys() != state["requires_grad"].keys():
        raise ValueError("model structure differs from the checkpoint")
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    for name, training in state["module_training"].items():
        modules[name].training = training
    for name, requires_grad in state["requires_grad"].items():
        params[name].requires_grad_(requires_grad)
    torch.set_num_threads(state["torch_threads"])
    torch.use_deterministic_algorithms(state["deterministic_algorithms"])
    random.setstate(state["python_rng"])
    numpy_rng = state["numpy_rng"]
    np.random.set_state((numpy_rng[0], numpy_rng[1].numpy().astype(np.uint32), *numpy_rng[2:]))
    torch.set_rng_state(state["torch_rng"])
    if cuda_rng is not None:
        torch.cuda.set_rng_state_all(cuda_rng)
    if generator is not None:
        generator.set_state(state["generator_rng"])
    return {"step": state["step"], "tokens_seen": state["tokens_seen"], "trainer_state": state["trainer_state"]}

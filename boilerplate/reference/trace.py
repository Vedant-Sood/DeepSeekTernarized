"""A reference trace: inputs, per-layer hidden states and logits from one forward pass.

Saved as ``<name>.safetensors`` (tensors) + ``<name>.json`` (metadata), so a trace produced on a GPU
box with the trusted runtime can be compared later against any Boilerplate mode on any machine.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


@dataclass
class ReferenceTrace:
    input_ids: torch.Tensor  # [b, s] int64
    hidden_states: list[torch.Tensor]  # one per layer boundary; index 0 = embeddings (if captured)
    logits: torch.Tensor | None  # [b, s, vocab] float32
    meta: dict = field(default_factory=dict)  # backend, model, dtype, hidden-state semantics, ...

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tensors = {"input_ids": self.input_ids.contiguous()}
        for i, h in enumerate(self.hidden_states):
            tensors[f"hidden.{i}"] = h.contiguous()
        if self.logits is not None:
            tensors["logits"] = self.logits.contiguous()
        save_file(tensors, str(path.with_suffix(".safetensors")))
        meta = {**self.meta, "num_hidden_states": len(self.hidden_states)}
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=str))

    @classmethod
    def load(cls, path: str | Path) -> "ReferenceTrace":
        path = Path(path)
        t = load_file(str(path.with_suffix(".safetensors")))
        meta = json.loads(path.with_suffix(".json").read_text())
        hs = [t[f"hidden.{i}"] for i in range(meta["num_hidden_states"])]
        return cls(t["input_ids"], hs, t.get("logits"), meta)

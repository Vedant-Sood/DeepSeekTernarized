"""Memory-mapped safetensors weights.

Every shard is mapped copy-on-write (``numpy.memmap`` mode ``"c"``), and each tensor is a zero-copy
torch view into that mapping, so nothing is read from disk until it is touched. Routed experts and
engram tables, which are most of the bytes, therefore stay on disk (in the OS page cache at best)
until a token actually routes to them; this is the engine's baseline "offload cold experts to disk"
mode. A tensor whose byte offset is not aligned for its dtype is copied once instead.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import torch

ST_DTYPES = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
    "F8_E8M0": torch.float8_e8m0fnu,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}


ST_TORCH = {str(v).replace("torch.", ""): v for v in ST_DTYPES.values()}
ST_TORCH["float4_e2m1fn_x2"] = torch.float4_e2m1fn_x2


class _Shard:
    def __init__(self, path: Path):
        self.path = path
        with path.open("rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(n))
        self.data_start = 8 + n
        self.meta = {k: v for k, v in header.items() if k != "__metadata__"}
        self._bytes: torch.Tensor | None = None

    @property
    def bytes(self) -> torch.Tensor:
        if self._bytes is None:
            mm = np.memmap(self.path, dtype=np.uint8, mode="c")
            self._bytes = torch.from_numpy(mm)
        return self._bytes

    def get(self, name: str) -> torch.Tensor:
        m = self.meta[name]
        b, e = m["data_offsets"]
        raw = self.bytes[self.data_start + b : self.data_start + e]
        shape = list(m["shape"])
        if m["dtype"] == "F4":  # logical shape, two values per byte along the last dim
            shape[-1] //= 2
            return raw.view(torch.uint8).view(shape).view(torch.float4_e2m1fn_x2)
        dtype = ST_DTYPES[m["dtype"]]
        size = torch.empty((), dtype=dtype).element_size()
        if (self.data_start + b) % size:
            raw = raw.clone()
        return raw.view(dtype).view(shape)


class WeightStore:
    """Name -> tensor over all shards of a checkpoint directory (zero-copy, lazy)."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        index = self.root / "model.safetensors.index.json"
        files = sorted({*json.loads(index.read_text())["weight_map"].values()}) if index.exists() else [
            p.name for p in sorted(self.root.glob("*.safetensors"))
        ]
        self.shards = [_Shard(self.root / f) for f in files]
        self.where: dict[str, _Shard] = {}
        for s in self.shards:
            for n in s.meta:
                self.where[n] = s
        # Boilerplate-format directories carry a manifest: per weight base name its mode
        # (A original, B entropy-coded, C stacked ternary) and how it is stored
        mpath = self.root / "manifest.json"
        self.manifest: dict | None = json.loads(mpath.read_text()) if mpath.exists() else None
        self.coded: dict[str, dict] = {}  # Mode B: original tensor name -> coding info
        if self.manifest:
            for meta in self.manifest["tensors"].values():
                if meta["mode"] == "B":
                    self.coded.update(meta["coded"])
                    self._mode_b = (meta["maxlen"], meta["block"])

    def mode_of(self, base: str) -> dict:
        """Manifest entry of a weight base name ({"mode": "A"} for plain checkpoints)."""
        if self.manifest is None:
            return {"mode": "A"}
        return self.manifest["tensors"].get(base, {"mode": "A"})

    def __contains__(self, name: str) -> bool:
        return name in self.where or name in self.coded

    def names(self) -> list[str]:
        return list(self.where)

    def get(self, name: str) -> torch.Tensor:
        if name in self.coded:  # Mode B: decode back to the exact original bytes
            from ..format import entropy

            info = self.coded[name]
            parts = {k: self.where[f"{name}.h{k}"].get(f"{name}.h{k}") for k in ("stream", "offsets", "lens")}
            if f"{name}.hraw" in self.where:
                parts["raw"] = self.get(f"{name}.hraw")
            maxlen, block = self._mode_b
            raw = entropy.decode(parts, info["symbols"], maxlen, block)
            return raw.view(ST_TORCH[info["dtype"]]).view(info["shape"])
        try:
            return self.where[name].get(name)
        except KeyError:
            raise KeyError(f"tensor {name!r} not in checkpoint {self.root}") from None

    def maybe(self, name: str) -> torch.Tensor | None:
        return self.get(name) if name in self else None

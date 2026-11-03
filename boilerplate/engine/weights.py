"""Memory-mapped safetensors weights.

Every shard is mapped copy-on-write (``numpy.memmap`` mode ``"c"``), and each tensor is a zero-copy
torch view into that mapping, so nothing is read from disk until it is touched. Routed experts and
engram tables, which are most of the bytes, therefore stay on disk (in the OS page cache at best)
until a token actually routes to them; this is the engine's baseline "offload cold experts to disk"
mode. A tensor whose byte offset is not aligned for its dtype is copied once instead.
"""

from __future__ import annotations

import json
import math
import struct
from collections import OrderedDict
from dataclasses import dataclass
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


@dataclass(frozen=True)
class TensorRef:
    """A coded tensor reference that does not retain its decoded bytes."""

    store: "WeightStore"
    name: str

    def get(self) -> torch.Tensor:
        return self.store.get(self.name)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.store.coded[self.name]["shape"])

    @property
    def dtype(self) -> torch.dtype:
        return ST_TORCH[self.store.coded[self.name]["dtype"]]


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

    def __init__(self, root: str | Path, mode_b_cache_bytes: int = 0):
        if mode_b_cache_bytes < 0:
            raise ValueError("mode_b_cache_bytes must be nonnegative")
        self.root = Path(root)
        self.mode_b_cache_bytes = mode_b_cache_bytes
        self.mode_b_cache_used = 0
        self.mode_b_cache_hits = 0
        self.mode_b_cache_misses = 0
        self.mode_b_cache_evictions = 0
        self._mode_b_cache: OrderedDict[tuple[str, int, int], torch.Tensor] = OrderedDict()
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
        self.coded_params: dict[str, tuple[int, int]] = {}
        if self.manifest:
            for meta in self.manifest["tensors"].values():
                if meta["mode"] == "B":
                    self.coded.update(meta["coded"])
                    for name in meta["coded"]:
                        self.coded_params[name] = (meta["maxlen"], meta["block"])

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
            info = self.coded[name]
            _, block = self.coded_params[name]
            nblocks = (info["symbols"] + block - 1) // block
            return self._decode_blocks(name, 0, nblocks).view(info["shape"])
        try:
            return self.where[name].get(name)
        except KeyError:
            raise KeyError(f"tensor {name!r} not in checkpoint {self.root}") from None

    def maybe(self, name: str) -> torch.Tensor | None:
        return self.get(name) if name in self else None

    def ref(self, name: str) -> torch.Tensor | TensorRef:
        """Return a lazy reference for coded tensors and a mapped tensor otherwise."""
        return TensorRef(self, name) if name in self.coded else self.get(name)

    def maybe_ref(self, name: str) -> torch.Tensor | TensorRef | None:
        return self.ref(name) if name in self else None

    def _decode_blocks(self, name: str, first: int, last: int) -> torch.Tensor:
        """Decode a contiguous range of independent Huffman blocks, caching original bytes."""
        from ..format import entropy

        key = (name, first, last)
        cached = self._mode_b_cache.get(key)
        if cached is not None:
            self._mode_b_cache.move_to_end(key)
            self.mode_b_cache_hits += 1
            return cached
        self.mode_b_cache_misses += 1
        info = self.coded[name]
        maxlen, block = self.coded_params[name]
        start = first * block
        end = min(last * block, info["symbols"])
        parts = {k: self.where[f"{name}.h{k}"].get(f"{name}.h{k}") for k in ("stream", "offsets", "lens")}
        parts["offsets"] = parts["offsets"][first:last + 1]
        if f"{name}.hraw" in self.where:
            parts["raw"] = self.where[f"{name}.hraw"].get(f"{name}.hraw")[start:end]
        raw = entropy.decode(parts, end - start, maxlen, block)
        tensor = raw.view(ST_TORCH[info["dtype"]])
        size = raw.numel()
        if size <= self.mode_b_cache_bytes:
            while self.mode_b_cache_used + size > self.mode_b_cache_bytes and self._mode_b_cache:
                _, old = self._mode_b_cache.popitem(last=False)
                self.mode_b_cache_used -= old.numel() * old.element_size()
                self.mode_b_cache_evictions += 1
            self._mode_b_cache[key] = tensor
            self.mode_b_cache_used += size
        return tensor

    def rows(self, name: str, indices: torch.Tensor) -> torch.Tensor:
        """Select rows without materializing a coded table; indices may repeat or be unsorted."""
        if name not in self.coded:
            t = self.get(name)
            return t.index_select(0, indices.reshape(-1)).view(*indices.shape, *t.shape[1:])
        info = self.coded[name]
        shape = info["shape"]
        if not shape:
            raise ValueError(f"cannot select rows from scalar tensor {name!r}")
        idx = indices.reshape(-1).tolist()
        if any(i < 0 or i >= shape[0] for i in idx):
            raise IndexError(f"row index out of bounds for {name!r}")
        out = torch.empty((len(idx), *shape[1:]), dtype=ST_TORCH[info["dtype"]])
        for i, row in enumerate(idx):
            out[i].copy_(self.row_range(name, row, row + 1)[0])
        return out.view(*indices.shape, *shape[1:])

    def row_range(self, name: str, start: int, end: int) -> torch.Tensor:
        """Select consecutive rows, decoding only their Huffman blocks when coded."""
        if name not in self.coded:
            return self.get(name)[start:end]
        info = self.coded[name]
        shape = info["shape"]
        if not shape or start < 0 or end < start or end > shape[0]:
            raise IndexError(f"invalid row range [{start}, {end}) for {name!r}")
        if start == end:
            return torch.empty((0, *shape[1:]), dtype=ST_TORCH[info["dtype"]])
        row_size = math.prod(shape[1:])
        first_symbol, end_symbol = start * row_size, end * row_size
        _, block = self.coded_params[name]
        first = first_symbol // block
        last = (end_symbol + block - 1) // block
        decoded = self._decode_blocks(name, first, last)
        lo = first_symbol - first * block
        return decoded[lo:lo + end_symbol - first_symbol].view(end - start, *shape[1:])

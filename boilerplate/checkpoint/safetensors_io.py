"""Read safetensors headers without loading tensor data.

A safetensors file starts with an 8-byte little-endian header length followed by a JSON header
mapping tensor names to ``{"dtype", "shape", "data_offsets"}``. Reading just the header tells us
every tensor's storage dtype and shape, which is what the config parser needs to verify the
checkpoint's real storage format (rather than trusting ``quantization_config``).
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path

# Bits per *stored element* for each safetensors dtype string.
# Note: safetensors records torch.float4_e2m1fn_x2 as "F4" with the *logical* (unpacked) shape.
DTYPE_BITS: dict[str, int] = {
    "F64": 64,
    "F32": 32,
    "F16": 16,
    "BF16": 16,
    "F8_E4M3": 8,
    "F8_E5M2": 8,
    "F8_E8M0": 8,
    "F4": 4,
    "I64": 64,
    "I32": 32,
    "I16": 16,
    "I8": 8,
    "U8": 8,
    "BOOL": 8,
}


@dataclass(frozen=True)
class TensorInfo:
    name: str
    dtype: str  # safetensors dtype string, e.g. "F8_E4M3"
    shape: tuple[int, ...]
    nbytes: int
    file: str | None = None

    @property
    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n


def parse_header_bytes(raw: bytes, file: str | None = None) -> dict[str, TensorInfo]:
    header = json.loads(raw)
    out: dict[str, TensorInfo] = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        start, end = meta["data_offsets"]
        out[name] = TensorInfo(name, meta["dtype"], tuple(meta["shape"]), end - start, file)
    return out


def read_header(path: str | Path) -> dict[str, TensorInfo]:
    path = Path(path)
    with path.open("rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        if n > 512 * 1024 * 1024:
            raise ValueError(f"{path}: implausible safetensors header length {n}")
        return parse_header_bytes(f.read(n), file=path.name)


def read_remote_headers(repo_id: str, revision: str | None = None) -> dict[str, TensorInfo]:
    """Fetch every shard header of a Hub repo with HTTP range requests (no weight download)."""
    from huggingface_hub import get_safetensors_metadata

    meta = get_safetensors_metadata(repo_id, revision=revision)
    out: dict[str, TensorInfo] = {}
    for file, fmeta in meta.files_metadata.items():
        for name, t in fmeta.tensors.items():
            start, end = t.data_offsets
            out[name] = TensorInfo(name, t.dtype, tuple(t.shape), end - start, file)
    return out

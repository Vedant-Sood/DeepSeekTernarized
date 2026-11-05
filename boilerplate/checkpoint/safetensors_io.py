"""Read safetensors headers without loading tensor data.

A safetensors file starts with an 8-byte little-endian header length followed by a JSON header
mapping tensor names to ``{"dtype", "shape", "data_offsets"}``. Reading just the header tells us
every tensor's storage dtype and shape, which is what the config parser needs to verify the
checkpoint's real storage format (rather than trusting ``quantization_config``).
"""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path

MAX_HEADER_BYTES = 100_000_000

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


def _unique_object(pairs: list[tuple]) -> dict:
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate safetensors header key: {key!r}")
        out[key] = value
    return out


def parse_header_metadata(raw: bytes, data_size: int | None = None) -> dict:
    """Validate metadata before any tensor view or allocation uses it."""
    if not 0 < len(raw) <= MAX_HEADER_BYTES or not raw.startswith(b"{"):
        raise ValueError("invalid safetensors header size or prefix")
    header = json.loads(raw, object_pairs_hook=_unique_object)
    if not isinstance(header, dict):
        raise ValueError("safetensors header must be an object")
    ranges = []
    for name, meta in header.items():
        if name == "__metadata__":
            if not isinstance(meta, dict) or any(not isinstance(v, str) for v in meta.values()):
                raise ValueError("safetensors metadata must contain strings")
            continue
        if not isinstance(meta, dict) or not isinstance(meta.get("dtype"), str) or meta["dtype"] not in DTYPE_BITS:
            raise ValueError(f"unsupported safetensors dtype for {name!r}")
        shape, offsets = meta.get("shape"), meta.get("data_offsets")
        if (not isinstance(shape, list) or any(type(d) is not int or d < 0 or d > 2**63 - 1 for d in shape) or
                not isinstance(offsets, list) or len(offsets) != 2 or
                any(type(v) is not int or v < 0 or v > 2**63 - 1 for v in offsets)):
            raise ValueError(f"invalid safetensors shape or offsets for {name!r}")
        start, end = offsets
        bits = math.prod(shape) * DTYPE_BITS[meta["dtype"]]
        if end < start or bits % 8 or end - start != bits // 8:
            raise ValueError(f"safetensors tensor size mismatch for {name!r}")
        if meta["dtype"] == "F4" and (not shape or shape[-1] % 2):
            raise ValueError("packed F4 requires an even final dimension")
        ranges.append((start, end))
    cursor = 0
    for start, end in sorted(ranges):
        if start != cursor:
            raise ValueError("safetensors data has a gap or overlap")
        cursor = end
    if data_size is not None and cursor != data_size:
        raise ValueError("safetensors offsets do not match the file size")
    return header


def parse_header_bytes(raw: bytes, file: str | None = None) -> dict[str, TensorInfo]:
    header = parse_header_metadata(raw)
    out: dict[str, TensorInfo] = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        start, end = meta["data_offsets"]
        out[name] = TensorInfo(name, meta["dtype"], tuple(meta["shape"]), end - start, file)
    return out


def read_header_metadata(path: str | Path) -> tuple[int, dict]:
    path = Path(path)
    with path.open("rb") as f:
        size = f.seek(0, 2)
        f.seek(0)
        prefix = f.read(8)
        if len(prefix) != 8:
            raise ValueError(f"{path}: truncated safetensors length")
        (n,) = struct.unpack("<Q", prefix)
        if not 0 < n <= MAX_HEADER_BYTES or n > size - 8:
            raise ValueError(f"{path}: implausible safetensors header length {n}")
        raw = f.read(n)
        if len(raw) != n:
            raise ValueError(f"{path}: truncated safetensors header")
        return 8 + n, parse_header_metadata(raw, data_size=size - 8 - n)


def read_header(path: str | Path) -> dict[str, TensorInfo]:
    _, header = read_header_metadata(path)
    return {
        name: TensorInfo(name, meta["dtype"], tuple(meta["shape"]),
                         meta["data_offsets"][1] - meta["data_offsets"][0], Path(path).name)
        for name, meta in header.items() if name != "__metadata__"
    }


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

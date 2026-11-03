"""Parameter / byte accounting by tensor role, computed from shard headers (exact, not estimated)."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from ..checkpoint.dequant import is_packed_fp4
from ..checkpoint.source import CheckpointSource
from . import roles


@dataclass
class RoleSize:
    tensors: int = 0
    params: int = 0  # logical weight elements (packed FP4 counted unpacked; scales excluded)
    weight_bytes: int = 0
    scale_bytes: int = 0

    @property
    def total_bytes(self) -> int:
        return self.weight_bytes + self.scale_bytes

    @property
    def bits_per_weight(self) -> float:
        return 8 * self.total_bytes / self.params if self.params else 0.0


def size_breakdown(src: CheckpointSource, include_mtp: bool = True) -> dict[str, RoleSize]:
    if src.headers is None:
        raise ValueError("size breakdown needs shard headers (open the checkpoint with read_headers=True)")
    h = src.headers
    out: dict[str, RoleSize] = defaultdict(RoleSize)
    for name, t in h.items():
        if not include_mtp and roles.is_mtp(name):
            continue
        key = ("mtp:" if roles.is_mtp(name) else "") + roles.classify(name)
        rs = out[key]
        if roles.is_scale(name):
            rs.scale_bytes += t.nbytes
            continue
        rs.tensors += 1
        rs.weight_bytes += t.nbytes
        numel = t.numel
        if t.dtype in ("I8", "U8"):
            sname = name[: -len("weight")] + "scale"
            if sname in h and h[sname].shape and is_packed_fp4(t.shape, h[sname].shape, t.shape[1] * 2 // max(1, h[sname].shape[-1])):
                numel *= 2
        rs.params += numel
    return dict(out)


def format_breakdown(sizes: dict[str, RoleSize]) -> str:
    total = RoleSize()
    lines = [f"{'role':<26}{'tensors':>9}{'params':>18}{'bytes':>18}{'bits/w':>9}{'share':>8}"]
    grand = sum(s.total_bytes for s in sizes.values()) or 1
    for role, s in sorted(sizes.items(), key=lambda kv: -kv[1].total_bytes):
        lines.append(
            f"{role:<26}{s.tensors:>9}{s.params:>18,}{s.total_bytes:>18,}{s.bits_per_weight:>9.3f}{100 * s.total_bytes / grand:>7.1f}%"
        )
        total.tensors += s.tensors
        total.params += s.params
        total.weight_bytes += s.weight_bytes
        total.scale_bytes += s.scale_bytes
    lines.append(f"{'TOTAL':<26}{total.tensors:>9}{total.params:>18,}{total.total_bytes:>18,}{total.bits_per_weight:>9.3f}{100.0:>7.1f}%")
    return "\n".join(lines)

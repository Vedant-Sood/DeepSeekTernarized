"""Tiered cache for routed experts: device ↔ host ↔ disk, with hit rates.

A MoE model touches a few experts per token, so most expert weights can stay cold. Each engine
builds an expert (its gate/up/down linears) on first use from the memory-mapped checkpoint
("disk": the OS page cache backs it). This cache bounds what stays materialized:

* **device tier** (GPU, when the engine runs there): experts moved with ``to_device``; byte budget
  ``device_bytes``. Evicted experts drop back to the host tier, not to disk.
* **host tier** (CPU RAM): built experts, including lazily materialized buffers such as Mode B
  decodes and dequantized ternary caches; byte budget ``host_bytes``. Evicted experts are
  dropped; the next use rebuilds them from the mmap'd files.

Sizes are measured from the tensors an expert actually holds (``resident_bytes``) after each use,
so a dequantization cache that appears during the first matmul is counted. ``stats()`` reports
hits and misses per tier, evictions and bytes promoted, for the logs the M5 runtime asks for.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Callable, Hashable

import torch


def resident_bytes(obj: Any, _seen: set | None = None) -> int:
    """Bytes of every distinct tensor reachable from ``obj`` (tuples, lists, dicts, attributes)."""
    seen = set() if _seen is None else _seen
    if isinstance(obj, torch.Tensor):
        # the tensor's own bytes, not its storage: mmap'd weights are views into whole shards
        key = (obj.data_ptr(), obj.numel(), obj.device)
        if key in seen:
            return 0
        seen.add(key)
        return obj.numel() * obj.element_size()
    if id(obj) in seen or obj is None or isinstance(obj, (int, float, str, bool, torch.dtype)):
        return 0
    seen.add(id(obj))
    if isinstance(obj, (tuple, list)):
        return sum(resident_bytes(x, seen) for x in obj)
    if isinstance(obj, dict):
        return sum(resident_bytes(x, seen) for x in obj.values())
    if hasattr(obj, "__dict__"):
        return sum(resident_bytes(x, seen) for x in vars(obj).values())
    return 0


class ExpertCache:
    def __init__(self, host_bytes: int | None = None, device_bytes: int = 0,
                 to_device: Callable[[Any], Any] | None = None, measure: Callable[[Any], int] = resident_bytes):
        """``host_bytes=None`` keeps every built expert (the old behaviour); 0 keeps none."""
        self.host_bytes, self.device_bytes = host_bytes, device_bytes
        self.to_device, self.measure = to_device, measure
        self.host: OrderedDict[Hashable, list] = OrderedDict()    # key -> [value, bytes]
        self.device: OrderedDict[Hashable, list] = OrderedDict()
        self.counts = {"device_hits": 0, "host_hits": 0, "misses": 0, "host_evictions": 0,
                       "device_evictions": 0, "promoted_bytes": 0}

    def get(self, key: Hashable, build: Callable[[], Any]) -> Any:
        if key in self.device:
            self.device.move_to_end(key)
            self.counts["device_hits"] += 1
            return self.device[key][0]
        if key in self.host:
            self.host.move_to_end(key)
            self.counts["host_hits"] += 1
            value = self.host[key][0]
        else:
            self.counts["misses"] += 1
            value = build()
            self.host[key] = [value, self.measure(value)]
        if self.to_device is not None and self.device_bytes > 0:
            moved = self.to_device(value)
            size = self.measure(moved)
            if size <= self.device_bytes:
                self.device[key] = [moved, size]
                self.counts["promoted_bytes"] += size
                self._evict(self.device, self.device_bytes, "device_evictions")
                return moved
        return value

    def touch(self, key: Hashable) -> None:
        """Re-measure after use (lazy buffers) and enforce the budgets."""
        for tier in (self.device, self.host):
            if key in tier:
                tier[key][1] = self.measure(tier[key][0])
        self._evict(self.device, self.device_bytes, "device_evictions")
        if self.host_bytes is not None:
            self._evict(self.host, self.host_bytes, "host_evictions")

    def _evict(self, tier: OrderedDict, budget: int, counter: str) -> None:
        while tier and sum(v[1] for v in tier.values()) > budget:
            tier.popitem(last=False)
            self.counts[counter] += 1

    def stats(self) -> dict:
        lookups = self.counts["device_hits"] + self.counts["host_hits"] + self.counts["misses"]
        return {**self.counts, "lookups": lookups,
                "hit_rate": (lookups - self.counts["misses"]) / lookups if lookups else None,
                "host_resident_bytes": sum(v[1] for v in self.host.values()),
                "device_resident_bytes": sum(v[1] for v in self.device.values()),
                "host_experts": len(self.host), "device_experts": len(self.device)}

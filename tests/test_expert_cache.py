"""Tiered expert cache: budgets, hit accounting, and identical engine outputs under any budget."""

import torch

from boilerplate.engine import Engine
from boilerplate.engine.expert_cache import ExpertCache, resident_bytes
from boilerplate.testing.tiny import write_tiny_checkpoint


def test_resident_bytes_counts_views_once():
    a = torch.zeros(10)
    assert resident_bytes((a, a, [a])) == 40
    assert resident_bytes({"x": a, "y": torch.zeros(2, dtype=torch.int8)}) == 42


def test_budgets_tiers_and_stats():
    built = []
    cache = ExpertCache(host_bytes=80, device_bytes=40, to_device=lambda v: v.clone())

    def build(i):
        built.append(i)
        return torch.full((10,), float(i))  # 40 bytes

    for key in (0, 1, 0, 2, 0):
        cache.get(key, lambda k=key: build(k))
        cache.touch(key)
    s = cache.stats()
    assert built == [0, 1, 2]                      # 0 stayed resident on host
    assert s["misses"] == 3 and s["lookups"] == 5
    assert s["device_experts"] == 1 and s["device_resident_bytes"] == 40
    assert s["host_resident_bytes"] <= 80 and s["host_evictions"] >= 1
    zero = ExpertCache(host_bytes=0)
    for _ in range(3):
        zero.get("e", lambda: torch.zeros(4))
        zero.touch("e")
    assert zero.stats()["misses"] == 3 and zero.stats()["host_experts"] == 0


def test_v4_engine_identical_under_zero_expert_budget(tmp_path):
    ckpt = write_tiny_checkpoint(tmp_path / "tiny")
    ids = torch.randint(0, 100, (1, 12), generator=torch.Generator().manual_seed(2))
    a = Engine.load(ckpt, max_seq_len=64).trace(ids, decode_from=8)
    eng = Engine.load(ckpt, max_seq_len=64, expert_cache_bytes=0)
    b = eng.trace(ids, decode_from=8)
    assert torch.equal(a.logits, b.logits)
    s = eng.model.expert_cache.stats()
    assert s["host_experts"] == 0 and s["misses"] == s["lookups"] > 0

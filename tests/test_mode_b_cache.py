"""Mode B block selection and bounded decoded-weight cache."""

import json

import pytest
import torch
from safetensors.torch import save_file

from boilerplate.engine import Engine
from boilerplate.engine.weights import TensorRef, WeightStore
from boilerplate.format import entropy
from boilerplate.format.export import export
from boilerplate.format.policy import Policy
from boilerplate.kernels.cpu import ext
from boilerplate.reference import compare_traces
from boilerplate.testing.tiny import write_tiny_checkpoint


pytestmark = pytest.mark.skipif(ext() is None, reason="C++ extension unavailable")


@pytest.fixture(scope="module")
def checkpoints(tmp_path_factory):
    root = tmp_path_factory.mktemp("mode_b_cache")
    source = write_tiny_checkpoint(root / "source")
    encoded = export(source, root / "encoded", Policy.from_dict({"default": {"mode": "B"}}), log=lambda *args: None)
    return source, encoded


def test_lru_hit_and_eviction(checkpoints):
    _, encoded = checkpoints
    ws = WeightStore(encoded, mode_b_cache_bytes=8192)
    name = "embed.weight"  # BF16: each 4096-symbol block decodes to 8192 bytes
    expected = ws.get(name).clone()
    ws.rows(name, torch.tensor([0]))
    misses = ws.mode_b_cache_misses
    ws.rows(name, torch.tensor([1]))  # same block
    assert ws.mode_b_cache_misses == misses
    assert ws.mode_b_cache_hits > 0
    assert torch.equal(ws.rows(name, torch.tensor([33])).view(torch.uint8), expected[33:34].view(torch.uint8))
    assert ws.mode_b_cache_evictions > 0
    assert ws.mode_b_cache_used <= ws.mode_b_cache_bytes


@pytest.mark.parametrize("dtype", [torch.uint8, torch.int8, torch.float4_e2m1fn_x2, torch.float8_e4m3fn,
                                   torch.float8_e8m0fnu, torch.bfloat16, torch.float16])
def test_rows_match_source_across_block_boundaries(tmp_path, dtype):
    generator = torch.Generator().manual_seed(7)
    raw = torch.randint(0, 128, (11, 1025), dtype=torch.uint8, generator=generator)
    tensor = raw.view(dtype) if dtype in (torch.int8, torch.float4_e2m1fn_x2) else raw
    if dtype not in (torch.int8, torch.uint8, torch.float4_e2m1fn_x2):
        tensor = (raw.float() / 128).to(dtype)
    parts = entropy.encode(tensor, block=1024)
    name = "matrix.weight"
    save_file({f"{name}.h{k}": v for k, v in parts.items()}, str(tmp_path / "weights.safetensors"))
    info = {"dtype": str(dtype).replace("torch.", ""), "shape": list(tensor.shape), "symbols": tensor.numel()}
    manifest = {"tensors": {"matrix": {"mode": "B", "coded": {name: info}, "maxlen": 12, "block": 1024}}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    ws = WeightStore(tmp_path, mode_b_cache_bytes=4096)

    indices = torch.tensor([[10, 0, 3], [4, 3, 7]])
    expected = tensor.index_select(0, indices.flatten()).view(*indices.shape, 1025)
    actual = ws.rows(name, indices)
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    assert torch.equal(ws.row_range(name, 2, 9).view(torch.uint8), tensor[2:9].view(torch.uint8))
    assert ws.row_range(name, 4, 4).shape == (0, 1025)
    assert ws.mode_b_cache_used <= ws.mode_b_cache_bytes
    with pytest.raises(IndexError):
        ws.rows(name, torch.tensor([-1]))


@pytest.mark.parametrize("budget", [0, 8192])
def test_bounded_engine_trace_matches_mode_a(checkpoints, budget):
    source, encoded = checkpoints
    ids = torch.randint(0, 512, (1, 20), generator=torch.Generator().manual_seed(13))
    baseline = Engine.load(source, max_seq_len=64).trace(ids, decode_from=7)
    engine = Engine.load(encoded, max_seq_len=64, mode_b_cache_bytes=budget)
    assert isinstance(engine.model.embed, TensorRef)
    assert isinstance(engine.model.head, TensorRef)
    assert engine.model.layers[1].engram.coded_table is not None
    actual = engine.trace(ids, decode_from=7)
    comparison = compare_traces(baseline, actual)
    assert all(layer.bit_exact for layer in comparison.layers), comparison.table()
    assert comparison.logits.bit_exact, comparison.table()
    assert engine.model.w.mode_b_cache_used <= budget
    experts = [entry[0] for entry in engine.model.expert_cache.host.values()]
    assert experts
    assert all(isinstance(linear._weight, TensorRef) for expert in experts for linear in expert)


def test_negative_cache_budget_is_rejected(checkpoints):
    with pytest.raises(ValueError, match="nonnegative"):
        WeightStore(checkpoints[1], mode_b_cache_bytes=-1)

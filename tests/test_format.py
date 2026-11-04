"""M2/M3: policies, export, manifest, Mode B lossless coding and Mode C loading on the tiny checkpoint."""

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from boilerplate.engine import Engine
from boilerplate.engine.weights import WeightStore
from boilerplate.format import entropy
from boilerplate.format.export import export
from boilerplate.format.policy import Policy, Rule
from boilerplate.kernels.cpu import ext
from boilerplate.reference import compare_traces
from boilerplate.testing.tiny import write_tiny_checkpoint

needs_ext = pytest.mark.skipif(ext() is None, reason="C++ extension unavailable")
EXPERTS = r"^layers\.\d+\.ffn\.experts\.\d+\.w[123]$"


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    return write_tiny_checkpoint(tmp_path_factory.mktemp("tiny"))


@pytest.fixture(scope="module")
def ids():
    torch.manual_seed(0)
    return torch.randint(0, 512, (1, 80))


@pytest.fixture(scope="module")
def mode_a(tiny, ids):
    return Engine.load(tiny, max_seq_len=128).trace(ids)


def test_policy_rules_and_bits():
    p = Policy.from_dict({"name": "x", "default": {"mode": "A"}, "rules": [
        {"match": EXPERTS, "mode": "C", "k": 2, "packing": "tq1"},
        {"match": r"engram\.embed$", "mode": "B"}]})
    assert p.rule_for("layers.3.ffn.experts.7.w2").mode == "C"
    assert p.rule_for("layers.1.engram.embed").mode == "B"
    assert p.rule_for("layers.3.ffn.gate").mode == "A"
    assert p.rules[0].effective_bits() == pytest.approx(3.25 + 0.2)  # 2 x (1.6 + 16/128)
    with pytest.raises(ValueError):
        Rule(match=".", mode="C", k=5)
    with pytest.raises(ValueError):
        Rule(match=".", mode="Z")
    mixed = Policy.from_dict({"name": "roundtrip", "default": {"mode": "B"}, "rules": [
        {"match": EXPERTS, "mode": "C", "k": 2, "packing": "tq1"}]})
    restored = Policy.from_dict(mixed.to_dict())
    assert restored.rule_for("layers.0.attn.wq_a").mode == "B"
    assert restored.rule_for("layers.0.ffn.experts.0.w1").mode == "C"
    assert restored.rule_for("layers.0.ffn.experts.0.w1").packing == "tq1"


@needs_ext
@pytest.mark.parametrize("dtype", [torch.uint8, torch.float8_e4m3fn, torch.bfloat16, torch.float16])
def test_entropy_roundtrip(dtype):
    torch.manual_seed(0)
    x = (torch.randn(10007) * 0.05).to(dtype) if dtype.is_floating_point else torch.randint(0, 16, (10007,), dtype=dtype)
    p = entropy.encode(x)
    n = entropy.split_bytes(x)[0].numel()
    back = entropy.decode(p, n).view(dtype)
    assert torch.equal(back.view(torch.uint8), x.view(torch.uint8))
    assert entropy.coded_bytes(p) < x.numel() * x.element_size()


@needs_ext
def test_entropy_length_limit():
    # geometric frequencies force Huffman lengths past maxlen; the limited code must still be prefix-free and decode
    freq = torch.tensor([2 ** min(i, 30) for i in range(40)] + [0] * 216)
    lens = entropy.code_lengths(freq, 12)
    assert int(lens.max()) <= 12
    assert sum(2.0 ** -int(l) for l in lens if l) <= 1.0
    sym = torch.repeat_interleave(torch.arange(40, dtype=torch.uint8), torch.tensor([1 + i % 7 for i in range(40)]))
    p = entropy.encode(sym)
    assert torch.equal(entropy.decode(p, sym.numel()), sym)


def test_size_guard_refuses_no_saving(tiny, tmp_path):
    # TQ2 k=2 is 4.25 b/w, the same as the FP4 experts: refused unless allow_larger
    bad = Policy.from_dict({"name": "bad", "rules": [{"match": EXPERTS, "mode": "C", "k": 2, "packing": "tq2"}]})
    with pytest.raises(ValueError, match="not below"):
        export(tiny, tmp_path / "bad", bad, log=lambda *a: None)


@needs_ext
def test_mode_b_bit_exact(tiny, ids, mode_a, tmp_path):
    out = export(tiny, tmp_path / "b", Policy.from_dict({"name": "lossless", "default": {"mode": "B"}}), log=lambda *a: None)
    man = json.loads((out / "manifest.json").read_text())
    assert man["summary"]["by_mode"]["B"]["bits_per_weight"] < 8
    src, dst = WeightStore(tiny), WeightStore(out)
    for n in src.names():
        assert torch.equal(src.get(n).view(torch.uint8), dst.get(n).view(torch.uint8)), n
    tr = Engine.load(out, max_seq_len=128).trace(ids)
    assert compare_traces(mode_a, tr).logits.bit_exact


@needs_ext
def test_mixed_mode_b_c_export_and_run(tiny, ids, tmp_path):
    policy_dir = Path(__file__).resolve().parents[1] / "configs" / "policy"
    policy = Policy.load(policy_dir / "experts_k2_lossless_rest.yaml")
    assert not policy.default.allow_larger and all(not rule.allow_larger for rule in policy.rules)
    mixed = export(tiny, tmp_path / "mixed", policy, log=lambda *a: None)
    c_only = export(tiny, tmp_path / "c_only", Policy.load(policy_dir / "experts_k2.yaml"), log=lambda *a: None)
    manifest = json.loads((mixed / "manifest.json").read_text())
    meta = manifest["tensors"]
    assert {entry["mode"] for entry in meta.values()} >= {"A", "B", "C"}
    recorded_policy = Policy.from_dict(manifest["policy"])
    assert recorded_policy.rule_for("layers.0.attn.wq_a").mode == "B"
    assert recorded_policy.rule_for("layers.0.ffn.experts.0.w1").mode == "C"
    src, dst = WeightStore(tiny), WeightStore(mixed)
    coded_names = [name for entry in meta.values() if entry["mode"] == "B" for name in entry["coded"]]
    assert coded_names
    b_names = [f"{base}.{suffix}" for base, entry in meta.items() if entry["mode"] == "B"
               for suffix in ("weight", "scale") if f"{base}.{suffix}" in src]
    assert set(coded_names) <= set(b_names)
    for name in b_names:
        assert torch.equal(src.get(name).view(torch.uint8), dst.get(name).view(torch.uint8)), name

    expected = Engine.load(c_only, max_seq_len=128).trace(ids, decode_from=13)
    actual = Engine.load(mixed, max_seq_len=128).trace(ids, decode_from=13)
    comparison = compare_traces(expected, actual)
    assert all(layer.bit_exact for layer in comparison.layers), comparison.table()
    assert comparison.logits.bit_exact, comparison.table()


def test_mode_b_keeps_wide_tensors_byte_identical(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text("{}")
    weight = torch.randn(8, 16, generator=torch.Generator().manual_seed(19), dtype=torch.float32)
    save_file({"linear.weight": weight}, str(source / "weights.safetensors"))
    out = export(source, tmp_path / "encoded", Policy.from_dict({"default": {"mode": "B"}}), log=lambda *a: None)
    meta = json.loads((out / "manifest.json").read_text())["tensors"]["linear"]
    assert meta["mode"] == "B" and meta["coded"] == {}
    assert torch.equal(WeightStore(out).get("linear.weight").view(torch.uint8), weight.view(torch.uint8))


@pytest.mark.parametrize("engram", [False, True])
def test_mode_c_export_and_run(tiny, ids, mode_a, tmp_path, engram):
    rules = [{"match": EXPERTS, "mode": "C", "k": 2, "packing": "tq1"}]
    if engram:
        rules.append({"match": r"^layers\.\d+\.engram\.embed$", "mode": "C", "k": 2, "packing": "tq1"})
    out = export(tiny, tmp_path / f"c{engram}", Policy.from_dict({"name": "c", "rules": rules}), log=lambda *a: None)
    man = json.loads((out / "manifest.json").read_text())
    t = man["tensors"]
    ex = next(b for b in t if "experts" in b and b.endswith("w1"))
    assert t[ex]["mode"] == "C" and t[ex]["packing"] == "tq1" and t[ex]["terms"] == 2
    assert (t["layers.1.engram.embed"]["mode"] == "C") == engram
    for meta in t.values():  # checksums cover every stored tensor
        assert all(len(s["sha256"]) == 64 and s["file"].startswith("model-") for s in meta["stored"].values())
    # the ternary weights are what was fitted from the source
    ws = WeightStore(out)
    assert ws.get(f"{ex}.tq0.packed").dtype == torch.uint8
    tr = Engine.load(out, max_seq_len=128).trace(ids)
    cmp = compare_traces(mode_a, tr)
    assert torch.isfinite(tr.logits).all()
    # tiny's weights are random Gaussians (the hardest case for ternary); this checks the path, not quality
    assert 0.5 < cmp.layers[-1].cos_mean < 1.0
    # the stored terms reconstruct the source expert to the error the fit promises (k=2 on Gaussians)
    from boilerplate.format.export import dequant_source
    from boilerplate.ternary.pack import unpack
    src = WeightStore(tiny)
    W = dequant_source(src.get(f"{ex}.weight"), src.get(f"{ex}.scale"))
    K = W.size(1)
    R = sum(unpack(ws.get(f"{ex}.tq{i}.packed"), "tq1", K).float() * ws.get(f"{ex}.tq{i}.scale").float().repeat_interleave(128, -1)[:, :K]
            for i in range(2))
    assert ((W - R).norm() / W.norm()) < 0.3

"""Ternary weights to llama.cpp TQ1_0/TQ2_0 GGUF tensors, exact round trip."""

import json

import numpy as np
import pytest
import torch
from safetensors.torch import load_file, save_file

from boilerplate.engine.weights import WeightStore
from boilerplate.format.export import export
from boilerplate.format.policy import Policy, Rule

gguf = pytest.importorskip("gguf")
transformers = pytest.importorskip("transformers")

PROJ = r"layers\.\d+\.(self_attn\.\w_proj|mlp\.(\w+_proj|experts\.\d+\.\w+_proj))$"


def _model(path, kind):
    torch.manual_seed(5)
    if kind == "llama":
        cfg = transformers.LlamaConfig(vocab_size=97, hidden_size=256, intermediate_size=512, num_hidden_layers=2,
                                       num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)
        model = transformers.LlamaForCausalLM(cfg)
    else:
        cfg = transformers.OlmoeConfig(vocab_size=97, hidden_size=256, intermediate_size=256, num_hidden_layers=2,
                                       num_attention_heads=4, num_key_value_heads=4, num_experts=3,
                                       num_experts_per_tok=2, max_position_embeddings=64)
        model = transformers.OlmoeForCausalLM(cfg)
    model.to(torch.bfloat16).save_pretrained(path, safe_serialization=True)
    if kind == "olmoe":  # published per-expert layout
        f = path / "model.safetensors"
        t = load_file(f)
        for i in range(cfg.num_hidden_layers):
            p = f"model.layers.{i}.mlp.experts"
            if p + ".gate_up_proj" in t:
                gu, down = t.pop(p + ".gate_up_proj"), t.pop(p + ".down_proj")
                for e in range(cfg.num_experts):
                    g, u = gu[e].chunk(2, 0)
                    t[f"{p}.{e}.gate_proj.weight"], t[f"{p}.{e}.up_proj.weight"] = g.contiguous(), u.contiguous()
                    t[f"{p}.{e}.down_proj.weight"] = down[e].contiguous()
        save_file(t, str(f))
    return cfg


def _export(tmp_path, kind, packing="tq1", group=256):
    src = tmp_path / "src"
    cfg = _model(src, kind)
    pol = Policy("gguf", [Rule(match=PROJ, mode="C", k=1, group=group, packing=packing, fit="absmean")])
    return cfg, export(src, tmp_path / "exp", pol, log=lambda *_: None)


@pytest.mark.parametrize("packing", ["tq1", "tq2"])
def test_llama_ternary_gguf_round_trip(tmp_path, packing):
    from boilerplate.format.gguf_export import llama_permute, write_ternary_only
    from boilerplate.ternary.pack import unpack

    cfg, exp = _export(tmp_path, "llama", packing)
    info = write_ternary_only(exp, tmp_path / "t.gguf", packing)
    assert len(info["converted"]) == 2 * 7 and not info["skipped"]
    reader = gguf.GGUFReader(str(tmp_path / "t.gguf"))
    tensors = {t.name: t for t in reader.tensors}
    store = WeightStore(exp)
    for name, base, kv in (("blk.0.attn_q.weight", "model.layers.0.self_attn.q_proj", 4),
                           ("blk.1.attn_k.weight", "model.layers.1.self_attn.k_proj", 2),
                           ("blk.0.ffn_down.weight", "model.layers.0.mlp.down_proj", None)):
        t = tensors[name]
        assert t.tensor_type.name == {"tq1": "TQ1_0", "tq2": "TQ2_0"}[packing]
        K = store.mode_of(base)["shape"][1]
        want = unpack(store.get(base + ".tq0.packed"), packing, K).float() * store.get(base + ".tq0.scale").float().repeat_interleave(256, -1)
        if kv is not None:
            want = llama_permute(want, 4, kv)
        got = gguf.quants.dequantize(t.data, t.tensor_type).reshape(want.shape)
        assert np.array_equal(got, want.numpy())


def test_olmoe_experts_are_stacked(tmp_path):
    from boilerplate.format.gguf_export import ternary_gguf_tensors

    cfg, exp = _export(tmp_path, "olmoe")
    _, values, skipped = ternary_gguf_tensors(exp)
    assert not skipped
    assert values["blk.0.ffn_gate_exps.weight"].shape == (3, 256, 256)
    assert "blk.1.attn_q.weight" in values


def test_incompatible_tensors_are_reported(tmp_path):
    from boilerplate.format.gguf_export import ternary_gguf_tensors

    _, exp = _export(tmp_path, "llama", group=128)
    _, values, skipped = ternary_gguf_tensors(exp)
    assert not values and all("group 128" in r for r in skipped.values())


def test_merge_into_base_swaps_only_ternary_tensors(tmp_path):
    from boilerplate.format.gguf_export import llama_permute, merge_into_base, ternary_gguf_tensors

    cfg, exp = _export(tmp_path, "llama")
    _, values, _ = ternary_gguf_tensors(exp)
    # stand-in for llama.cpp's BF16 conversion: same names and shapes, F32 data, some metadata
    base = tmp_path / "base.gguf"
    w = gguf.GGUFWriter(str(base), "llama")
    w.add_block_count(cfg.num_hidden_layers)
    w.add_string("tokenizer.ggml.model", "gpt2")
    for name, v in values.items():
        w.add_tensor(name, np.zeros(tuple(v.shape), dtype=np.float32))
    w.add_tensor("token_embd.weight", np.ones((97, 256), dtype=np.float32))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    info = merge_into_base(base, exp, tmp_path / "merged.gguf", "tq2")
    assert set(info["converted"]) == set(values)
    reader = gguf.GGUFReader(str(tmp_path / "merged.gguf"))
    assert reader.fields["tokenizer.ggml.model"].contents() == "gpt2"
    out = {t.name: t for t in reader.tensors}
    assert out["token_embd.weight"].tensor_type.name == "F32"
    q = out["blk.0.attn_q.weight"]
    assert q.tensor_type.name == "TQ2_0"
    assert np.array_equal(gguf.quants.dequantize(q.data, q.tensor_type).reshape(256, 256), values["blk.0.attn_q.weight"].numpy())

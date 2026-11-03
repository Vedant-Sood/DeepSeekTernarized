import json
from pathlib import Path

import pytest

from boilerplate.checkpoint import CheckpointSource
from boilerplate.config import SpecError, parse_checkpoint
from boilerplate.config import roles
from boilerplate.config.report import size_breakdown
from boilerplate.testing.tiny import TINY_TEXT_CONFIG, tensor_names, tiny_config, write_tiny_checkpoint

FIXTURES = Path(__file__).parent / "fixtures"
REAL_CONFIG = json.loads((FIXTURES / "deepseek_v41_flash_1017_config.json").read_text())


@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory):
    return write_tiny_checkpoint(tmp_path_factory.mktemp("tiny"))


def test_tiny_spec_matches_config(tiny_dir):
    spec = parse_checkpoint(CheckpointSource.open(tiny_dir))
    t = TINY_TEXT_CONFIG
    assert spec.warnings == []
    assert spec.family == "deepseek_v4"
    assert (spec.num_layers, spec.hidden_size, spec.vocab_size) == (t["num_hidden_layers"], t["hidden_size"], t["vocab_size"])
    att = spec.attention
    assert att.type == "deepseek_v4_sparse" and att.attn_sink
    assert [l.compress_ratio for l in att.layers] == t["compress_ratios"]
    assert [l.kv_source_layer for l in att.layers] == [None, None, 2, 2, 4, 4, 4, 4]
    assert [l.index_source_layer for l in att.layers] == [None, None, 2, 2, 4, 4, 6, 6]
    assert [l.rope for l in att.layers] == ["main"] * 2 + ["compress"] * 6
    assert spec.rope.scaling_type is None and spec.rope.theta == t["rope_theta"]
    assert spec.rope_compress.scaling_type == "yarn" and spec.rope_compress.theta == t["compress_rope_theta"]
    moe = spec.moe
    assert (moe.n_routed_experts, moe.n_shared_experts, moe.top_k) == (8, 1, 2)
    assert moe.router_bias and not moe.router_bias_vl
    assert moe.moe_layers == list(range(8)) and moe.dense_layers == []
    st = spec.storage
    assert st.verified_from_headers
    assert st.role_formats["routed_expert"] == {"fp4_e2m1/group32/F8_E8M0": 8 * 3 * 8}
    assert set(st.role_formats["attention"]) == {"fp8_e4m3/block32x32/F8_E8M0"}
    assert set(st.role_formats["embedding"]) == {"bf16"}
    assert spec.hyper_connections["hc_mult"] == 4


def test_size_breakdown_counts_fp4_unpacked(tiny_dir):
    sizes = size_breakdown(CheckpointSource.open(tiny_dir))
    t = TINY_TEXT_CONFIG
    experts = sizes["routed_expert"]
    assert experts.params == t["num_hidden_layers"] * t["n_routed_experts"] * 3 * t["hidden_size"] * t["moe_intermediate_size"]
    assert experts.bits_per_weight == pytest.approx(4.25)  # 4-bit e2m1 + one 8-bit scale per 32
    assert sizes["attention"].bits_per_weight == pytest.approx(8 + 8 / 1024, rel=1e-3)


def test_no_headers_falls_back_to_declared(tiny_dir):
    spec = parse_checkpoint(CheckpointSource.open(tiny_dir, read_headers=False))
    assert not spec.storage.verified_from_headers
    assert spec.storage.role_formats["routed_expert"] == {"fp4(declared)": 8 * 8 * 3}
    assert any("headers not read" in w for w in spec.warnings)


def test_index_inconsistencies_become_warnings(tmp_path):
    root = write_tiny_checkpoint(tmp_path / "c", config=tiny_config(n_routed_experts=8))
    index_path = root / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    # drop expert 7 of layer 2 and the compressor of layer 4
    index["weight_map"] = {
        k: v for k, v in index["weight_map"].items()
        if not k.startswith("layers.2.ffn.experts.7.") and not k.startswith("layers.4.attn.compressor.")
    }
    index_path.write_text(json.dumps(index))
    src = CheckpointSource.open(root)
    src.headers = {k: v for k, v in src.headers.items() if k in src.weight_map}
    spec = parse_checkpoint(src)
    text = "\n".join(spec.warnings)
    assert "{2: 7}" in text
    assert "kv_source_layer_ids [2, 4] != layers with compressor tensors [2]" in text


def test_real_v41_config_structure():
    """The released config.json + its tensor-name layout (no weights) parse into the right plan."""
    src = CheckpointSource("hub-names-only", REAL_CONFIG, {n: "x" for n in tensor_names(REAL_CONFIG)}, headers=None)
    spec = parse_checkpoint(src)
    assert spec.architecture == "DeepseekV41ForCausalLM" and spec.family == "deepseek_v4"
    assert (spec.num_layers, spec.hidden_size, spec.vocab_size) == (40, 5120, 129280)
    att = spec.attention
    assert (att.num_heads, att.num_kv_heads, att.head_dim, att.qk_rope_head_dim) == (64, 1, 512, 64)
    assert att.sliding_window == 128 and att.indexer["topk"] == 512 and att.indexer["candidate_source_layer"] == 20
    by = {l.layer: l for l in att.layers}
    assert by[0].kind == by[1].kind == "sliding" and by[39].kind == "sliding+compressed"
    assert (by[2].kv_source_layer, by[7].kv_source_layer, by[13].kv_source_layer, by[19].kv_source_layer) == (2, 2, 8, 14)
    assert (by[20].compress_ratio, by[20].kv_source_layer, by[39].compress_ratio, by[39].kv_source_layer) == (1, 20, 1, 20)
    assert (by[25].index_source_layer, by[37].index_source_layer) == (24, 36)
    assert att.indexer["mtp_compress_ratios"] == [0, 0, 0]
    assert spec.moe.n_routed_experts == 384 and spec.moe.top_k == 6 and spec.moe.routed_scaling_factor == 1.5
    assert spec.moe.scoring_func == "sqrtsoftplus" and spec.moe.topk_method == "noaux_tc"
    assert spec.rope_compress.factor == 16 and spec.rope_compress.original_max_position == 65536
    assert spec.rope_compress.theta == 160000
    assert spec.engram["layer_ids"] == [1, 14] and spec.engram["head_dim"] == 256
    assert spec.mtp["num_layers"] == 3 and spec.mtp["dspark"]["dspark_n_routed_experts"] == 128
    assert spec.vision["config"]["num_hidden_layers"] == 32
    assert spec.storage.declared["expert_dtype"] == "fp4" and spec.storage.declared["weight_block_size"] == [32, 32]
    assert spec.special_tokens["image_token_id"] == 129264


def test_missing_required_field_raises():
    cfg = tiny_config()
    del cfg["text_config"]["hidden_size"]
    with pytest.raises(SpecError):
        parse_checkpoint(CheckpointSource("x", cfg, {}, None))


@pytest.mark.parametrize(
    "name,role",
    [
        ("layers.3.ffn.experts.17.w2.weight", "routed_expert"),
        ("layers.3.ffn.experts.17.w2.scale", "routed_expert"),
        ("layers.3.ffn.shared_experts.w1.weight", "shared_expert"),
        ("layers.3.ffn.gate.weight", "router"),
        ("layers.3.ffn.gate.bias", "router_bias"),
        ("layers.3.ffn.gate.bias_vl", "router_bias"),
        ("layers.3.attn.wq_b.weight", "attention"),
        ("layers.3.attn.q_norm.weight", "norm"),
        ("layers.3.attn.compressor.norm.weight", "attn_compressor"),
        ("layers.3.attn.indexer.weights_proj.weight", "attn_indexer"),
        ("layers.3.attn.attn_sink", "attn_sink"),
        ("layers.3.hc_attn_fn", "hyper_connection"),
        ("layers.14.engram.embed.weight", "engram_table"),
        ("layers.14.engram.wkv.weight", "engram"),
        ("embed.weight", "embedding"),
        ("head.weight", "lm_head"),
        ("norm.weight", "norm"),
        ("mtp.0.ffn.experts.3.w1.weight", "routed_expert"),
        ("vision.blocks.0.attn.wqkv.weight", "vision"),
        # HF-style V3 naming
        ("model.layers.5.mlp.experts.12.gate_proj.weight", "routed_expert"),
        ("model.layers.5.mlp.gate.e_score_correction_bias", "router_bias"),
        ("model.layers.0.mlp.down_proj.weight", "dense_mlp"),
        ("model.layers.0.self_attn.kv_a_proj_with_mqa.weight", "attention"),
        ("model.layers.0.input_layernorm.weight", "norm"),
        ("model.embed_tokens.weight", "embedding"),
        ("lm_head.weight", "lm_head"),
    ],
)
def test_roles(name, role):
    assert roles.classify(name) == role

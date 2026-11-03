"""Write a tiny random-init checkpoint in the DeepSeek V4.1 Flash on-disk format.

The output mirrors the real release: HF-style ``config.json`` (with ``text_config`` and
``quantization_config``), ``model.safetensors.index.json``, several shards, and the same tensor
names and storage formats -- FP8 e4m3 linears with 32x32 ue8m0 block scales, routed experts as packed
FP4 e2m1 (int8 ``[out, in/2]``) with ue8m0 scales per 32 inputs, BF16 embeddings / head / small
projections, FP32 hyper-connection, attention-sink and router-bias tensors.

Tensor names and module structure follow the official reference ``inference/model.py``. Dtypes of
unscaled tensors that the reference casts on load (e.g. compressor weights) are a best guess; the
config parser never relies on them -- it reads whatever the headers say.

Engram, vision and DSpark/MTP stages are left out of the tiny config for now (engram needs the
reference ``engram.py`` hashing and a tokenizer); see docs/reports/M0.md.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import torch
from safetensors.torch import save_file

from ..checkpoint.dequant import quant_fp4_group, quant_fp8_block

TINY_TEXT_CONFIG = {
    "model_type": "deepseek_v41_text",
    "vocab_size": 512,
    "hidden_size": 128,
    "moe_intermediate_size": 64,
    "num_hidden_layers": 6,
    "num_attention_heads": 4,
    "num_key_value_heads": 1,
    "head_dim": 64,
    "qk_rope_head_dim": 16,
    "q_lora_rank": 64,
    "o_lora_rank": 32,
    "o_groups": 2,
    "hidden_act": "silu",
    "swiglu_limit": 10.0,
    "rms_norm_eps": 1e-20,
    "attention_bias": False,
    "tie_word_embeddings": False,
    "max_position_embeddings": 4096,
    "rope_theta": 10000,
    "rope_scaling": {"rope_type": "yarn", "factor": 4, "beta_fast": 32, "beta_slow": 1, "original_max_position_embeddings": 64},
    "n_routed_experts": 8,
    "n_shared_experts": 1,
    "num_experts_per_tok": 2,
    "scoring_func": "sqrtsoftplus",
    "topk_method": "noaux_tc",
    "norm_topk_prob": True,
    "routed_scaling_factor": 1.5,
    "sliding_window": 8,
    "compress_ratios": [0, 2, 2, 1, 1, 0],
    "compress_rope_theta": 40000,
    "kv_source_layer_ids": [1, 3],
    "index_source_layer_ids": [1, 3],
    "index_n_heads": 2,
    "index_head_dim": 32,
    "index_topk": 8,
    "candidate_source_layer_id": 1,
    "candidate_topk_blocks": 2,
    "candidate_block_size": 2,
    "hc_mult": 4,
    "hc_sinkhorn_iters": 20,
    "hc_eps": 1e-6,
    "engram_layer_ids": [],
    "num_nextn_predict_layers": 0,
}

TINY_CONFIG = {
    "architectures": ["DeepseekV41ForCausalLM"],
    "model_type": "deepseek_v41",
    "dtype": "bfloat16",
    "bos_token_id": 0,
    "eos_token_id": 1,
    "pad_token_id": 2,
    "quantization_config": {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "weight_block_size": [32, 32],
        "scale_fmt": "ue8m0",
        "expert_dtype": "fp4",
    },
    "text_config": TINY_TEXT_CONFIG,
}


def tiny_config(**overrides) -> dict:
    cfg = copy.deepcopy(TINY_CONFIG)
    cfg["text_config"].update(overrides)
    return cfg


def plan_tensors(config: dict) -> list[tuple[str, str, tuple[int, ...]]]:
    """Every tensor of the checkpoint as (name, kind, logical shape). Kinds: fp8 / fp4 (scaled
    linears, a ``.scale`` sibling is added on materialization), bf16, ones (bf16 norm weights),
    fp32, hc_scale."""
    t = {**config, **config.get("text_config", {})}
    fp4_experts = (config.get("quantization_config") or {}).get("expert_dtype") == "fp4"
    dim, n_layers, vocab = t["hidden_size"], t["num_hidden_layers"], t["vocab_size"]
    nh, hd, rd = t["num_attention_heads"], t["head_dim"], t["qk_rope_head_dim"]
    qr, orank, og = t["q_lora_rank"], t["o_lora_rank"], t["o_groups"]
    inter, n_exp = t["moe_intermediate_size"], t["n_routed_experts"]
    hc = t["hc_mult"]
    mix_hc = (2 + hc) * hc
    ratios = t["compress_ratios"]
    kv_src, idx_src = set(t["kv_source_layer_ids"]), set(t["index_source_layer_ids"])
    ih, ihd = t["index_n_heads"], t["index_head_dim"]
    assert rd <= hd
    plan: list[tuple[str, str, tuple[int, ...]]] = [
        ("embed.weight", "embed", (vocab, dim)),
        ("head.weight", "bf16", (vocab, dim)),
        ("norm.weight", "ones", (dim,)),
    ]
    add = lambda name, kind, *shape: plan.append((name, kind, tuple(shape)))  # noqa: E731
    for i in range(n_layers):
        p = f"layers.{i}"
        add(f"{p}.attn_norm.weight", "ones", dim)
        add(f"{p}.ffn_norm.weight", "ones", dim)
        for sub in ("attn", "ffn"):
            add(f"{p}.hc_{sub}_fn", "hc_fn", mix_hc, hc * dim)
            add(f"{p}.hc_{sub}_base", "hc_base", mix_hc)
            add(f"{p}.hc_{sub}_scale", "hc_scale", 3)
        a = f"{p}.attn"
        add(f"{a}.attn_sink", "sink", nh)
        add(f"{a}.wq_a", "fp8", qr, dim)
        add(f"{a}.q_norm.weight", "ones", qr)
        add(f"{a}.wq_b", "fp8", nh * hd, qr)
        add(f"{a}.wkv", "fp8", hd, dim)
        add(f"{a}.kv_norm.weight", "ones", hd)
        add(f"{a}.wo_a", "fp8", og * orank, nh * hd // og)
        add(f"{a}.wo_b", "fp8", dim, og * orank)
        if i in kv_src:
            add(f"{a}.compressor.wkv.weight", "bf16", hd, dim)
            if ratios[i] > 1:
                add(f"{a}.compressor.wgate.weight", "bf16", hd, dim)
            add(f"{a}.compressor.norm.weight", "ones", hd)
        if i in idx_src:
            add(f"{a}.indexer.wq_b", "fp8", ih * ihd, qr)
            add(f"{a}.indexer.weights_proj.weight", "bf16", ih, dim)
            if i in kv_src:
                add(f"{a}.indexer.wk.weight", "bf16", ihd, hd)
                add(f"{a}.indexer.k_norm.weight", "ones", ihd)
        f = f"{p}.ffn"
        add(f"{f}.gate.weight", "bf16", n_exp, dim)
        add(f"{f}.gate.bias", "router_bias", n_exp)
        ek = "fp4" if fp4_experts else "fp8"
        for e in range(n_exp):
            add(f"{f}.experts.{e}.w1", ek, inter, dim)
            add(f"{f}.experts.{e}.w2", ek, dim, inter)
            add(f"{f}.experts.{e}.w3", ek, inter, dim)
        for w, (o, i_) in {"w1": (inter, dim), "w2": (dim, inter), "w3": (inter, dim)}.items():
            add(f"{f}.shared_experts.{w}", "fp8", o, i_)
    return plan


def tensor_names(config: dict) -> list[str]:
    """Names as they appear in the index (scaled linears contribute .weight and .scale)."""
    out = []
    for name, kind, _ in plan_tensors(config):
        out += [f"{name}.weight", f"{name}.scale"] if kind in ("fp8", "fp4") else [name]
    return out


def build_tiny_state_dict(config: dict, seed: int = 0) -> dict[str, torch.Tensor]:
    block = tuple((config.get("quantization_config") or {}).get("weight_block_size", (32, 32)))
    g = torch.Generator().manual_seed(seed)

    def randn(*shape, std=1.0):
        return torch.randn(*shape, generator=g) * std

    sd: dict[str, torch.Tensor] = {}
    for name, kind, shape in plan_tensors(config):
        fan_in = shape[-1]
        if kind == "fp8":
            sd[f"{name}.weight"], sd[f"{name}.scale"] = quant_fp8_block(randn(*shape, std=fan_in**-0.5), block)
        elif kind == "fp4":
            sd[f"{name}.weight"], sd[f"{name}.scale"] = quant_fp4_group(randn(*shape, std=fan_in**-0.5), 32)
        elif kind == "embed":
            sd[name] = randn(*shape).bfloat16()
        elif kind == "bf16":
            sd[name] = randn(*shape, std=fan_in**-0.5).bfloat16()
        elif kind == "ones":
            sd[name] = torch.ones(*shape).bfloat16()
        elif kind == "hc_fn":
            sd[name] = randn(*shape, std=0.02)
        elif kind == "hc_base":
            sd[name] = randn(*shape, std=0.1)
        elif kind == "hc_scale":
            sd[name] = torch.ones(*shape) + randn(*shape, std=0.05)
        elif kind == "sink":
            sd[name] = randn(*shape, std=0.5)
        elif kind == "router_bias":
            sd[name] = randn(*shape, std=0.1)
        else:
            raise ValueError(kind)
    return sd


def write_tiny_checkpoint(out_dir: str | Path, config: dict | None = None, seed: int = 0, n_shards: int = 3) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    config = config or tiny_config()
    sd = build_tiny_state_dict(config, seed)
    names = list(sd)
    per = math.ceil(len(names) / n_shards)
    weight_map, total = {}, 0
    for k in range(n_shards):
        chunk = names[k * per : (k + 1) * per]
        if not chunk:
            continue
        fname = f"model-{k + 1:05d}-of-{n_shards:05d}.safetensors"
        save_file({n: sd[n].contiguous() for n in chunk}, str(out / fname))
        for n in chunk:
            weight_map[n] = fname
            total += sd[n].numel() * sd[n].element_size()
    (out / "config.json").write_text(json.dumps(config, indent=2))
    (out / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2)
    )
    return out

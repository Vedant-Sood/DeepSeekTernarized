"""Engine-side model config for the DeepSeek V4.x sparse-attention family.

Built from the checkpoint's own ``config.json`` (nothing about the shape is hard-coded). Field names
follow the official reference's ``ModelArgs`` because the HF -> reference key mapping is already
verified against the ``inference/config.json`` that ships with the release (see
``boilerplate.reference.official_backend.hf_config_to_model_args``); the engine adds the derived
per-layer plan (which layer's compressed KV / indexer each layer reads).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ..reference.official_backend import hf_config_to_model_args


@dataclass
class EngineConfig:
    vocab_size: int
    dim: int
    n_layers: int
    n_heads: int
    head_dim: int
    rope_head_dim: int
    q_lora_rank: int
    o_lora_rank: int
    o_groups: int
    norm_eps: float
    window_size: int
    compress_ratios: tuple[int, ...]
    kv_source_layers: tuple[int, ...]
    index_source_layers: tuple[int, ...]
    rope_theta: float
    compress_rope_theta: float
    rope_factor: float
    original_seq_len: int
    beta_fast: float
    beta_slow: float
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    candidate_source_layer: int
    candidate_topk_blocks: int
    candidate_block_size: int
    hc_mult: int
    hc_sinkhorn_iters: int
    hc_eps: float
    n_routed_experts: int
    n_activated_experts: int
    n_shared_experts: int
    moe_inter_dim: int
    score_func: str
    norm_topk_prob: bool
    route_scale: float
    swiglu_limit: float
    gate_temp: float = 1.0
    weight_dtype: str = "fp8"  # storage of the big linears, from quantization_config
    expert_dtype: str | None = "fp4"
    fp8_block: int = 32  # activation quant block; weight scale blocks are inferred from shapes
    engram: dict = field(default_factory=dict)  # raw engram_* keys (hashing needs them verbatim)
    eos_token_id: int | list[int] | None = None
    bos_token_id: int | None = None
    max_position_embeddings: int = 0
    raw: dict = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------ derived per-layer plan
    def kv_source_of(self, layer: int) -> int | None:
        """The layer whose compressed KV this layer attends over (None for window-only layers)."""
        if not self.compress_ratios[layer]:
            return None
        srcs = [s for s in self.kv_source_layers if s <= layer]
        if not srcs:
            raise ValueError(f"layer {layer} compresses KV but no kv source precedes it")
        src = max(srcs)
        if self.compress_ratios[src] != self.compress_ratios[layer]:
            raise ValueError(f"layer {layer} (ratio {self.compress_ratios[layer]}) reads layer {src}'s KV (ratio {self.compress_ratios[src]})")
        return src

    def index_source_of(self, layer: int) -> int | None:
        if not self.compress_ratios[layer]:
            return None
        srcs = [s for s in self.index_source_layers if s <= layer]
        return max(srcs) if srcs else None

    @property
    def mix_hc(self) -> int:
        return (2 + self.hc_mult) * self.hc_mult

    @classmethod
    def from_hf(cls, config: dict) -> "EngineConfig":
        a = hf_config_to_model_args(config, text_only=True)
        text = {**{k: v for k, v in config.items() if k != "text_config"}, **config.get("text_config", {})}
        n = a["n_layers"]
        ratios = tuple(a.get("compress_ratios") or (0,) * n)[:n]  # trailing entries belong to MTP stages
        q = config.get("quantization_config") or {}
        block = (q.get("weight_block_size") or [32, 32])[1]
        return cls(
            vocab_size=a["vocab_size"],
            dim=a["dim"],
            n_layers=n,
            n_heads=a["n_heads"],
            head_dim=a["head_dim"],
            rope_head_dim=a["rope_head_dim"],
            q_lora_rank=a["q_lora_rank"],
            o_lora_rank=a["o_lora_rank"],
            o_groups=a["o_groups"],
            norm_eps=a["norm_eps"],
            window_size=a["window_size"],
            compress_ratios=ratios,
            kv_source_layers=tuple(a.get("kv_source_layers") or ()),
            index_source_layers=tuple(a.get("index_source_layers") or ()),
            rope_theta=a["rope_theta"],
            compress_rope_theta=a.get("compress_rope_theta", a["rope_theta"]),
            rope_factor=a.get("rope_factor", 1),
            original_seq_len=a.get("original_seq_len", 0),
            beta_fast=a.get("beta_fast", 32),
            beta_slow=a.get("beta_slow", 1),
            index_n_heads=a.get("index_n_heads", 0),
            index_head_dim=a.get("index_head_dim", 0),
            index_topk=a.get("index_topk", 0),
            candidate_source_layer=a.get("candidate_source_layer", -1) if a.get("candidate_source_layer") is not None else -1,
            candidate_topk_blocks=a.get("candidate_topk_blocks", 0),
            candidate_block_size=a.get("candidate_block_size", 0),
            hc_mult=a["hc_mult"],
            hc_sinkhorn_iters=a["hc_sinkhorn_iters"],
            hc_eps=a["hc_eps"],
            n_routed_experts=a["n_routed_experts"],
            n_activated_experts=a["n_activated_experts"],
            n_shared_experts=a.get("n_shared_experts", 1),
            moe_inter_dim=a["moe_inter_dim"],
            score_func=a["score_func"],
            norm_topk_prob=a.get("norm_topk_prob", True),
            route_scale=a.get("route_scale", 1.0),
            swiglu_limit=a.get("swiglu_limit") or 0.0,
            gate_temp=a.get("gate_temp", 1.0),
            weight_dtype=a.get("dtype", "bf16"),
            expert_dtype=a.get("expert_dtype"),
            fp8_block=block,
            engram={k: v for k, v in text.items() if k.startswith("engram_")},
            eos_token_id=config.get("eos_token_id", text.get("eos_token_id")),
            bos_token_id=config.get("bos_token_id", text.get("bos_token_id")),
            max_position_embeddings=text.get("max_position_embeddings", 0),
            raw=config,
        )

    @classmethod
    def from_dir(cls, path: str | Path) -> "EngineConfig":
        return cls.from_hf(json.loads((Path(path) / "config.json").read_text()))

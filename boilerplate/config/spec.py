"""Architecture description derived from a checkpoint. Plain dataclasses, JSON-serializable."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class RopeSpec:
    head_dim: int  # rotary dims per head
    theta: float
    scaling_type: str | None  # e.g. "yarn", None for plain RoPE
    factor: float | None = None
    original_max_position: int | None = None
    beta_fast: float | None = None
    beta_slow: float | None = None
    max_position: int | None = None


@dataclass
class LayerAttentionSpec:
    """Per-layer attention plan. For the V4 sparse family: every layer attends over a sliding window;
    a layer with compress_ratio r > 0 also attends over r-to-1 compressed KV, produced by
    ``kv_source_layer`` and top-k selected by ``index_source_layer``'s indexer."""

    layer: int
    kind: str  # "sliding" | "sliding+compressed" | "mla" | "gqa" | "mha"
    compress_ratio: int = 0
    kv_source_layer: int | None = None
    index_source_layer: int | None = None
    rope: str = "main"  # which RopeSpec applies: "main" or "compress"


@dataclass
class AttentionSpec:
    type: str  # "deepseek_v4_sparse" | "mla" | "gqa" | "mha"
    num_heads: int
    num_kv_heads: int
    head_dim: int
    q_lora_rank: int | None = None
    kv_lora_rank: int | None = None
    o_lora_rank: int | None = None
    o_groups: int | None = None
    qk_rope_head_dim: int | None = None
    qk_nope_head_dim: int | None = None
    v_head_dim: int | None = None
    sliding_window: int | None = None
    attn_sink: bool = False
    indexer: dict | None = None  # n_heads, head_dim, topk, candidate prefilter
    layers: list[LayerAttentionSpec] = field(default_factory=list)


@dataclass
class MoESpec:
    n_routed_experts: int
    n_shared_experts: int
    top_k: int
    moe_intermediate_size: int
    scoring_func: str
    topk_method: str | None
    norm_topk_prob: bool
    routed_scaling_factor: float
    router_bias: bool  # selection-only correction bias present in checkpoint
    router_bias_vl: bool  # separate bias for image tokens
    swiglu_limit: float | None
    n_group: int | None = None
    topk_group: int | None = None
    moe_layers: list[int] = field(default_factory=list)
    dense_layers: list[int] = field(default_factory=list)
    dense_intermediate_size: int | None = None


@dataclass
class StorageSpec:
    """How the checkpoint actually stores weights, verified from shard headers where available."""

    quant_method: str | None
    declared: dict  # quantization_config as written in config.json
    formats: dict[str, int] = field(default_factory=dict)  # format name -> number of weight tensors
    role_formats: dict[str, dict[str, int]] = field(default_factory=dict)  # role -> format -> count
    verified_from_headers: bool = False


@dataclass
class ModelSpec:
    source: str
    architecture: str  # e.g. "DeepseekV41ForCausalLM"
    model_type: str
    family: str  # "deepseek_v4" | "deepseek_v3" | "generic"
    num_layers: int
    hidden_size: int
    vocab_size: int
    norm_eps: float
    hidden_act: str
    tie_word_embeddings: bool
    attention: AttentionSpec
    moe: MoESpec | None
    rope: RopeSpec
    rope_compress: RopeSpec | None
    storage: StorageSpec
    hyper_connections: dict | None = None
    engram: dict | None = None
    mtp: dict | None = None
    vision: dict | None = None
    special_tokens: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

"""Derive a :class:`ModelSpec` from a checkpoint's config.json, safetensors index and shard headers.

Nothing about the architecture is hardcoded: counts and dims come from config.json, and are then
cross-checked against what the tensor names (index) and shapes/dtypes (headers) actually contain.
Disagreements become ``spec.warnings`` (or a :class:`SpecError` when they make the spec unusable).

Field *semantics* are family specific (e.g. how DeepSeek V4.x shares compressed KV across layers);
those interpretations follow the official reference implementation shipped with the checkpoint and
are documented next to the code that applies them.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict

from ..checkpoint.dequant import infer_block, is_packed_fp4
from ..checkpoint.source import CheckpointSource
from . import roles
from .spec import AttentionSpec, LayerAttentionSpec, ModelSpec, MoESpec, RopeSpec, StorageSpec


class SpecError(ValueError):
    pass


_MISSING = object()


class _Cfg:
    """config.json view: text_config (if present) layered over the top-level dict, with aliases."""

    def __init__(self, config: dict):
        self.top = config
        self.text = {**config, **config.get("text_config", {})}
        self.used: set[str] = set()

    def get(self, *names: str, default=_MISSING):
        for n in names:
            if n in self.text and self.text[n] is not None:
                self.used.add(n)
                return self.text[n]
        if default is _MISSING:
            raise SpecError(f"config.json is missing required field (any of): {', '.join(names)}")
        return default

    def has(self, *names: str) -> bool:
        return any(n in self.text and self.text[n] is not None for n in names)


def _weight_format(name: str, src: CheckpointSource, declared: dict) -> str:
    """Storage format of one weight tensor, e.g. 'fp8_e4m3/block32x32/F8_E8M0'."""
    scale_name = None
    for suf in ("scale", "weight_scale_inv", "weight_scale"):
        cand = name[: -len("weight")] + suf if name.endswith("weight") else name + "." + suf
        if cand in src.weight_map:
            scale_name = cand
            break
    h = src.headers
    if h is None or name not in h:
        if scale_name is None:
            return "unverified"
        role = roles.classify(name)
        if role == "routed_expert" and declared.get("expert_dtype"):
            return f"{declared['expert_dtype']}(declared)"
        return f"{declared.get('quant_method', 'quantized')}(declared)"
    t = h[name]
    if scale_name is None or scale_name not in h:
        return {"BF16": "bf16", "F16": "fp16", "F32": "fp32"}.get(t.dtype, t.dtype.lower() + "/unscaled")
    s = h[scale_name]
    if t.dtype == "F8_E4M3":
        bo, bi = infer_block(t.shape, s.shape)
        return f"fp8_e4m3/block{bo}x{bi}/{s.dtype}"
    if t.dtype in ("I8", "U8") and is_packed_fp4(t.shape, s.shape, group=t.shape[1] * 2 // s.shape[1]):
        return f"fp4_e2m1/group{t.shape[1] * 2 // s.shape[1]}/{s.dtype}"
    if t.dtype == "F4":  # logical shape recorded by safetensors
        return f"fp4_e2m1/group{t.shape[1] // s.shape[1]}/{s.dtype}"
    return f"{t.dtype.lower()}/scaled-{s.dtype}"


def _logical_shape(name: str, src: CheckpointSource) -> tuple[int, ...] | None:
    """Unpacked [out, in] shape (packed FP4 int8 has half the stored columns)."""
    if src.headers is None or name not in src.headers:
        return None
    t = src.headers[name]
    if t.dtype in ("I8", "U8"):
        scale = name[: -len("weight")] + "scale"
        if scale in src.headers and is_packed_fp4(t.shape, src.headers[scale].shape, t.shape[1] * 2 // max(1, src.headers[scale].shape[1])):
            return (t.shape[0], t.shape[1] * 2)
    return t.shape


def parse_checkpoint(src: CheckpointSource) -> ModelSpec:
    cfg = _Cfg(src.config)
    warnings: list[str] = []
    names = src.tensor_names

    # ---------------------------------------------------------------- index survey
    backbone_layers: set[int] = set()
    experts_per_layer: dict[int, int] = defaultdict(int)
    mtp_experts: dict[int, int] = defaultdict(int)
    mtp_stages: set[int] = set()
    dense_mlp_layers: set[int] = set()
    compressor_layers: set[int] = set()
    indexer_layers: set[int] = set()
    role_names: dict[str, list[str]] = defaultdict(list)
    for n in names:
        role = roles.classify(n)
        role_names[role].append(n)
        if roles.is_mtp(n):
            m = re.match(r"(?:model\.)?mtp\.(\d+)\.", n)
            if m:
                mtp_stages.add(int(m.group(1)))
                e = roles.expert_index(n)
                if e is not None:
                    mtp_experts[int(m.group(1))] = max(mtp_experts[int(m.group(1))], e + 1)
            continue
        li = roles.layer_index(n)
        if li is None or role == "vision":
            continue
        backbone_layers.add(li)
        if role == "routed_expert":
            experts_per_layer[li] = max(experts_per_layer[li], roles.expert_index(n) + 1)
        elif role == "dense_mlp":
            dense_mlp_layers.add(li)
        elif role == "attn_compressor":
            compressor_layers.add(li)
        elif role == "attn_indexer":
            indexer_layers.add(li)

    # ---------------------------------------------------------------- core dims
    num_layers = int(cfg.get("num_hidden_layers", "n_layers"))
    hidden = int(cfg.get("hidden_size", "dim"))
    vocab = int(cfg.get("vocab_size"))
    if backbone_layers and max(backbone_layers) + 1 != num_layers:
        warnings.append(f"config num_hidden_layers={num_layers} but index has layers 0..{max(backbone_layers)}")
    if backbone_layers and len(backbone_layers) != max(backbone_layers) + 1:
        missing = sorted(set(range(max(backbone_layers) + 1)) - backbone_layers)
        warnings.append(f"index is missing layers {missing}")

    embed_name = next(iter(role_names.get("embedding", [])), None)
    if embed_name and (shape := _logical_shape(embed_name, src)):
        if tuple(shape) != (vocab, hidden):
            warnings.append(f"{embed_name} shape {shape} != (vocab_size, hidden_size)=({vocab}, {hidden})")

    is_v4 = cfg.has("compress_ratios", "hc_mult", "layer_types") and cfg.has("o_lora_rank", "hc_mult")
    is_mla = cfg.has("kv_lora_rank")
    family = "deepseek_v4" if is_v4 else ("deepseek_v3" if is_mla else "generic")

    # ---------------------------------------------------------------- RoPE
    n_heads = int(cfg.get("num_attention_heads", "n_heads"))
    head_dim = int(cfg.get("head_dim", default=hidden // n_heads))
    rope_dim = int(cfg.get("qk_rope_head_dim", "rope_head_dim", default=head_dim))
    scaling = cfg.get("rope_scaling", default=None) or {}
    rope_params = cfg.get("rope_parameters", default=None) or {}
    theta = float(cfg.get("rope_theta", default=rope_params.get("rope_theta", 10000.0)))
    max_pos = cfg.get("max_position_embeddings", default=None)

    def _rope(base: float, with_scaling: bool) -> RopeSpec:
        sc = scaling if with_scaling else {}
        stype = sc.get("rope_type", sc.get("type")) if sc else None
        return RopeSpec(
            head_dim=rope_dim,
            theta=base,
            scaling_type=None if stype in (None, "default") else stype,
            factor=sc.get("factor"),
            original_max_position=sc.get("original_max_position_embeddings"),
            beta_fast=sc.get("beta_fast"),
            beta_slow=sc.get("beta_slow"),
            max_position=max_pos,
        )

    rope_compress = None
    if is_v4:
        # Reference (inference/model.py, Attention.__init__): sliding-window-only layers use plain
        # RoPE at rope_theta with YaRN disabled; compressed layers use YaRN at compress_rope_theta.
        rope = _rope(theta, with_scaling=False)
        rope_compress = _rope(float(cfg.get("compress_rope_theta", default=theta)), with_scaling=True)
    else:
        rope = _rope(theta, with_scaling=True)

    # ---------------------------------------------------------------- attention
    n_kv = int(cfg.get("num_key_value_heads", default=n_heads))
    if is_v4:
        ratios = list(cfg.get("compress_ratios", default=[]))
        if not ratios:
            raise SpecError("deepseek_v4 family config without compress_ratios is not supported yet")
        n_mtp = int(cfg.get("num_nextn_predict_layers", "n_mtp_layers", default=0))
        if len(ratios) not in (num_layers, num_layers + n_mtp):
            warnings.append(f"compress_ratios has {len(ratios)} entries; expected {num_layers} (+{n_mtp} MTP)")
        compressing = [i for i in range(num_layers) if ratios[i] > 0]
        # Reference: only kv_source_layers own a Compressor; other compressing layers read the cache of
        # the most recent source. Same for index_source_layers and the Indexer. Without these lists,
        # every compressing layer is its own source.
        kv_sources = sorted(cfg.get("kv_source_layer_ids", "kv_source_layers", default=compressing))
        idx_sources = sorted(cfg.get("index_source_layer_ids", "index_source_layers", default=compressing))
        if compressor_layers and set(kv_sources) != compressor_layers:
            warnings.append(f"kv_source_layer_ids {kv_sources} != layers with compressor tensors {sorted(compressor_layers)}")
        if indexer_layers and set(idx_sources) != indexer_layers:
            warnings.append(f"index_source_layer_ids {idx_sources} != layers with indexer tensors {sorted(indexer_layers)}")

        def _latest(sources: list[int], i: int) -> int | None:
            prior = [s for s in sources if s <= i]
            return prior[-1] if prior else None

        layer_specs = []
        for i in range(num_layers):
            r = ratios[i]
            if r == 0:
                layer_specs.append(LayerAttentionSpec(i, "sliding", 0, rope="main"))
                continue
            kv_src, idx_src = _latest(kv_sources, i), _latest(idx_sources, i)
            if kv_src is None or idx_src is None:
                warnings.append(f"layer {i} compresses (ratio {r}) but has no preceding kv/index source")
            elif ratios[kv_src] != r:
                warnings.append(f"layer {i} ratio {r} reads compressed KV of layer {kv_src} with ratio {ratios[kv_src]}")
            layer_specs.append(LayerAttentionSpec(i, "sliding+compressed", r, kv_src, idx_src, rope="compress"))
        cand = int(cfg.get("candidate_source_layer_id", "candidate_source_layer", default=-1))
        attention = AttentionSpec(
            type="deepseek_v4_sparse",
            num_heads=n_heads,
            num_kv_heads=n_kv,
            head_dim=head_dim,
            q_lora_rank=cfg.get("q_lora_rank", default=None),
            o_lora_rank=cfg.get("o_lora_rank", default=None),
            o_groups=cfg.get("o_groups", default=None),
            qk_rope_head_dim=rope_dim,
            qk_nope_head_dim=head_dim - rope_dim,
            v_head_dim=head_dim,
            sliding_window=cfg.get("sliding_window", "window_size", default=None),
            attn_sink=bool(role_names.get("attn_sink")),
            indexer={
                "n_heads": cfg.get("index_n_heads", default=None),
                "head_dim": cfg.get("index_head_dim", default=None),
                "topk": cfg.get("index_topk", default=None),
                "candidate_source_layer": cand if cand >= 0 else None,
                "candidate_topk_blocks": cfg.get("candidate_topk_blocks", default=None),
                "candidate_block_size": cfg.get("candidate_block_size", default=None),
                "kv_source_layers": kv_sources,
                "index_source_layers": idx_sources,
                "mtp_compress_ratios": ratios[num_layers:],
            },
            layers=layer_specs,
        )
    elif is_mla:
        attention = AttentionSpec(
            type="mla",
            num_heads=n_heads,
            num_kv_heads=n_kv,
            head_dim=int(cfg.get("qk_nope_head_dim")) + rope_dim,
            q_lora_rank=cfg.get("q_lora_rank", default=None),
            kv_lora_rank=int(cfg.get("kv_lora_rank")),
            qk_rope_head_dim=rope_dim,
            qk_nope_head_dim=int(cfg.get("qk_nope_head_dim")),
            v_head_dim=int(cfg.get("v_head_dim")),
            layers=[LayerAttentionSpec(i, "mla") for i in range(num_layers)],
        )
    else:
        kind = "mha" if n_kv == n_heads else "gqa"
        attention = AttentionSpec(
            type=kind,
            num_heads=n_heads,
            num_kv_heads=n_kv,
            head_dim=head_dim,
            sliding_window=cfg.get("sliding_window", default=None),
            layers=[LayerAttentionSpec(i, kind) for i in range(num_layers)],
        )

    # ---------------------------------------------------------------- MoE
    moe = None
    if cfg.has("n_routed_experts", "num_experts", "num_local_experts"):
        n_routed = int(cfg.get("n_routed_experts", "num_experts", "num_local_experts"))
        moe_layers = sorted(experts_per_layer)
        bad = {li: n for li, n in experts_per_layer.items() if n != n_routed}
        if bad:
            warnings.append(f"layers with expert count != n_routed_experts={n_routed}: {bad}")
        if not moe_layers:
            warnings.append("config declares routed experts but the index has none")
        inter = int(cfg.get("moe_intermediate_size", "moe_inter_dim"))
        sample = next((n for n in role_names.get("routed_expert", []) if re.search(r"\.(w1|gate_proj)\.weight$", n) and not roles.is_mtp(n)), None)
        if sample and (shape := _logical_shape(sample, src)) and tuple(shape) != (inter, hidden):
            warnings.append(f"{sample} logical shape {shape} != (moe_intermediate_size, hidden)=({inter}, {hidden})")
        rb = role_names.get("router_bias", [])
        dense = sorted(dense_mlp_layers)
        fkd = cfg.get("first_k_dense_replace", default=None)
        if fkd is not None and dense and dense != list(range(int(fkd))):
            warnings.append(f"first_k_dense_replace={fkd} but dense MLP tensors at layers {dense}")
        moe = MoESpec(
            n_routed_experts=n_routed,
            n_shared_experts=int(cfg.get("n_shared_experts", default=0)),
            top_k=int(cfg.get("num_experts_per_tok", "n_activated_experts")),
            moe_intermediate_size=inter,
            scoring_func=str(cfg.get("scoring_func", "score_func", default="softmax")),
            topk_method=cfg.get("topk_method", default=None),
            norm_topk_prob=bool(cfg.get("norm_topk_prob", default=False)),
            routed_scaling_factor=float(cfg.get("routed_scaling_factor", "route_scale", default=1.0)),
            router_bias=any(n.endswith((".bias", "e_score_correction_bias")) for n in rb),
            router_bias_vl=any(n.endswith(".bias_vl") for n in rb),
            swiglu_limit=cfg.get("swiglu_limit", default=None),
            n_group=cfg.get("n_group", default=None),
            topk_group=cfg.get("topk_group", default=None),
            moe_layers=moe_layers,
            dense_layers=dense,
            dense_intermediate_size=cfg.get("intermediate_size", default=None) if dense else None,
        )

    # ---------------------------------------------------------------- storage
    declared = dict(src.config.get("quantization_config") or {})
    formats: Counter = Counter()
    role_formats: dict[str, Counter] = defaultdict(Counter)
    for n in names:
        if roles.is_scale(n):
            continue
        fmt = _weight_format(n, src, declared)
        formats[fmt] += 1
        role_formats[roles.classify(n)][fmt] += 1
    if src.headers is not None and declared.get("weight_block_size"):
        bo, bi = declared["weight_block_size"]
        observed = {f for f in formats if f.startswith("fp8_e4m3/block")}
        if observed and observed != {f"fp8_e4m3/block{bo}x{bi}/F8_E8M0"} and observed != {f"fp8_e4m3/block{bo}x{bi}/F32"}:
            warnings.append(f"declared weight_block_size {declared['weight_block_size']} but observed {sorted(observed)}")
    if src.headers is None:
        warnings.append("shard headers not read: storage formats are inferred from names + quantization_config")
    storage = StorageSpec(
        quant_method=declared.get("quant_method"),
        declared=declared,
        formats=dict(formats),
        role_formats={r: dict(c) for r, c in role_formats.items()},
        verified_from_headers=src.headers is not None,
    )

    # ---------------------------------------------------------------- extras
    hc = None
    if cfg.has("hc_mult"):
        hc = {"hc_mult": cfg.get("hc_mult"), "sinkhorn_iters": cfg.get("hc_sinkhorn_iters", default=None), "eps": cfg.get("hc_eps", default=None)}
    engram = None
    if cfg.get("engram_layer_ids", default=None):
        tables = {}
        for n in role_names.get("engram_table", []):
            if n.endswith(".weight") and (shape := _logical_shape(n, src)):
                tables[roles.layer_index(n)] = list(shape)
        engram = {
            "layer_ids": cfg.get("engram_layer_ids"),
            "num_embeddings": cfg.get("engram_num_embeddings", default=None),
            "max_ngram_size": cfg.get("engram_max_ngram_size", default=None),
            "vocab_size": cfg.get("engram_vocab_size", default=None),
            "n_heads": cfg.get("engram_n_heads", default=None),
            "head_dim": cfg.get("engram_head_dim", default=None),
            "pad_token_id": cfg.get("engram_pad_token_id", "engram_pad_id", default=None),
            "compressed_vocab_size": cfg.get("engram_compressed_vocab_size", default=None),
            "table_shapes": tables,
        }
        for li, shape in tables.items():
            k = engram["layer_ids"].index(li) if li in engram["layer_ids"] else None
            if k is not None and engram["num_embeddings"] and shape[0] < engram["num_embeddings"][k]:
                warnings.append(f"engram table at layer {li} has {shape[0]} rows < declared {engram['num_embeddings'][k]}")
    mtp = None
    n_mtp_cfg = int(cfg.get("num_nextn_predict_layers", "n_mtp_layers", default=0))
    if n_mtp_cfg or mtp_stages:
        if mtp_stages and len(mtp_stages) != n_mtp_cfg:
            warnings.append(f"num_nextn_predict_layers={n_mtp_cfg} but index has mtp stages {sorted(mtp_stages)}")
        mtp = {
            "num_layers": n_mtp_cfg,
            "stages_in_index": sorted(mtp_stages),
            "experts_per_stage": dict(mtp_experts),
            "dspark": {k: v for k, v in cfg.text.items() if k.startswith("dspark_")} or None,
        }
    vision = None
    if src.config.get("vision_config") or role_names.get("vision"):
        vision = {"config": src.config.get("vision_config"), "num_tensors": len(role_names.get("vision", []))}
    if role_names.get("other"):
        warnings.append(f"{len(role_names['other'])} tensors with unrecognised role, e.g. {role_names['other'][:5]}")

    return ModelSpec(
        source=src.location,
        architecture=(src.config.get("architectures") or ["?"])[0],
        model_type=str(src.config.get("model_type", "?")),
        family=family,
        num_layers=num_layers,
        hidden_size=hidden,
        vocab_size=vocab,
        norm_eps=float(cfg.get("rms_norm_eps", "norm_eps", default=1e-6)),
        hidden_act=str(cfg.get("hidden_act", default="silu")),
        tie_word_embeddings=bool(cfg.get("tie_word_embeddings", default=False)),
        attention=attention,
        moe=moe,
        rope=rope,
        rope_compress=rope_compress,
        storage=storage,
        hyper_connections=hc,
        engram=engram,
        mtp=mtp,
        vision=vision,
        special_tokens={k: src.config.get(k, cfg.text.get(k)) for k in ("bos_token_id", "eos_token_id", "pad_token_id", "image_token_id") if cfg.text.get(k) is not None},
        warnings=warnings,
    )

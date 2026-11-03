"""Classify checkpoint tensors into roles from their names.

Roles drive the mixed-precision policy (which tensors may be ternarized) and the size report.
Patterns cover both the DeepSeek V4.x naming (``layers.N.ffn.experts.E.w1``) and the HF-style
V2/V3 naming (``model.layers.N.mlp.experts.E.gate_proj``). The first matching rule wins, so more
specific rules come first. Roles ignore the ``mtp.`` prefix (see :func:`is_mtp`). Unknown names get the role ``other`` and are reported, never dropped.
"""

from __future__ import annotations

import re

# Scale tensors belong to the weight they scale; their role is the weight's role.
SCALE_SUFFIXES = (".scale", ".weight_scale_inv", ".weight_scale")

ROLE_RULES: list[tuple[str, str]] = [
    ("vision", r"^(model\.)?(vision|aligner|visual|vision_tower|multi_modal_projector)\.|^(model\.)?image_(start|end|newline)"),
    ("engram_table", r"\.engram\.embed\."),
    ("engram", r"\.engram\."),
    ("router_bias", r"\.(gate\.(bias|bias_vl)|gate\.e_score_correction_bias)$"),
    ("router", r"\.(ffn|mlp)\.gate\.weight$"),
    ("shared_expert", r"\.(ffn|mlp)\.shared_experts?\."),
    ("routed_expert", r"\.(ffn|mlp)\.experts\.\d+\."),
    ("attn_indexer", r"\.(attn|self_attn)\.indexer\."),
    ("attn_compressor", r"\.(attn|self_attn)\.compressor\."),
    ("attn_sink", r"\.(attn|self_attn)\.attn_sink$"),
    ("norm", r"(norm|layernorm)(\.weight)?$"),
    ("attention", r"\.(attn|self_attn)\."),
    ("dense_mlp", r"\.(ffn|mlp)\.(w[123]|gate_proj|up_proj|down_proj)\."),
    ("hyper_connection", r"\.hc_[a-z_]+$|^(model\.)?hc_"),
    ("embedding", r"^(model\.)?(embed|embed_tokens)\.weight$"),
    ("lm_head", r"^(model\.)?(head|lm_head)\.weight$"),
]
_COMPILED = [(role, re.compile(p)) for role, p in ROLE_RULES]

LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")
EXPERT_RE = re.compile(r"\.experts\.(\d+)\.")


def base_weight_name(name: str) -> str:
    for suf in SCALE_SUFFIXES:
        if name.endswith(suf):
            return name[: -len(suf)] + ".weight"
    return name


def is_scale(name: str) -> bool:
    return name.endswith(SCALE_SUFFIXES)


def classify(name: str) -> str:
    target = base_weight_name(name)
    for role, rx in _COMPILED:
        if rx.search(target):
            return role
    return "other"


def is_mtp(name: str) -> bool:
    """Multi-token-prediction / draft (DSpark) modules live under ``mtp.``; their role is still
    classified by the rest of the name, so policy rules can treat MTP experts like any others."""
    return name.startswith(("mtp.", "model.mtp."))


def layer_index(name: str) -> int | None:
    m = LAYER_RE.search(name)
    return int(m.group(1)) if m else None


def expert_index(name: str) -> int | None:
    m = EXPERT_RE.search(name)
    return int(m.group(1)) if m else None

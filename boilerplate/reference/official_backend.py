"""Reference backend: the official DeepSeek V4.x implementation shipped in the checkpoint repo.

No mainstream runtime supports ``deepseek_v41`` (transformers has ``deepseek_v4`` only, a different
architecture), so the trusted reference for V4.1 is the checkpoint's own ``inference/`` code. This
backend imports it unmodified from a local directory and:

* optionally swaps its TileLang ``kernel`` module for :mod:`.cpu_kernels` so it runs without a GPU;
* converts the HF checkpoint with the reference's own ``convert.py`` (model-parallel = 1);
* builds ``ModelArgs`` from the HF ``config.json`` (the mapping below is checked against the
  ``inference/config.json`` that ships next to the release);
* records every Block's output hidden state (the full ``[b, s, hc_mult, dim]`` residual stream) and
  full-sequence logits.

Get the code with:
    huggingface-cli download deepseek-ai/DeepSeek-V4.1-Flash-1017 --include "inference/*" --local-dir ref
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import torch

from . import cpu_kernels
from .trace import ReferenceTrace

# HF text_config key -> reference ModelArgs field (identity for keys not listed).
HF_TO_MODEL_ARGS = {
    "hidden_size": "dim",
    "moe_intermediate_size": "moe_inter_dim",
    "num_hidden_layers": "n_layers",
    "num_nextn_predict_layers": "n_mtp_layers",
    "num_attention_heads": "n_heads",
    "num_experts_per_tok": "n_activated_experts",
    "scoring_func": "score_func",
    "routed_scaling_factor": "route_scale",
    "qk_rope_head_dim": "rope_head_dim",
    "rms_norm_eps": "norm_eps",
    "sliding_window": "window_size",
    "kv_source_layer_ids": "kv_source_layers",
    "index_source_layer_ids": "index_source_layers",
    "candidate_source_layer_id": "candidate_source_layer",
    "engram_pad_token_id": "engram_pad_id",
    "dspark_num_experts_per_tok": "dspark_n_activated_experts",
}
VISION_TO_MODEL_ARGS = {
    "num_hidden_layers": "vision_n_layers",
    "hidden_size": "vision_dim",
    "num_attention_heads": "vision_n_heads",
    "intermediate_size": "vision_inter_dim",
    "patch_size": "vision_patch_size",
    "rope_theta": "vision_rope_theta",
    "downsample_ratio": "vision_downsample_ratio",
    "max_image_tokens": "vision_max_n_token",
    "min_pixels": "vision_min_pixels",
    "max_wh_ratio": "vision_max_wh_ratio",
}


def hf_config_to_model_args(config: dict, text_only: bool = False) -> dict:
    """Translate an HF-style V4.1 config.json into reference ``ModelArgs`` keyword arguments."""
    text = {**{k: v for k, v in config.items() if k != "text_config"}, **config.get("text_config", {})}
    out: dict = {}
    for k, v in text.items():
        if k == "rope_scaling" and v:
            out["rope_factor"] = v.get("factor", 1)
            out["original_seq_len"] = v.get("original_max_position_embeddings", 0)
            for b in ("beta_fast", "beta_slow"):
                if b in v:
                    out[b] = v[b]
        else:
            out[HF_TO_MODEL_ARGS.get(k, k)] = v
    q = config.get("quantization_config") or {}
    out["dtype"] = "fp8" if q.get("quant_method") == "fp8" else "bf16"
    out["expert_dtype"] = q.get("expert_dtype")
    vision = config.get("vision_config")
    if vision and not text_only:
        for k, v in vision.items():
            if k in VISION_TO_MODEL_ARGS:
                out[VISION_TO_MODEL_ARGS[k]] = v
    else:
        out["vision_n_layers"] = 0
    for k in ("compress_ratios", "kv_source_layers", "index_source_layers", "engram_layer_ids", "engram_num_embeddings", "dspark_target_layer_ids"):
        if isinstance(out.get(k), list):
            out[k] = tuple(out[k])
    return out


@contextmanager
def _isolated_import(code_dir: Path, use_cpu_kernels: bool):
    """Import the reference package's flat modules (model, kernel, engram, ...) without leaking them."""
    names = sorted({p.stem for p in code_dir.glob("*.py")} | {"kernel", "model"})
    saved = {n: sys.modules.pop(n) for n in list(sys.modules) if n in names}
    sys.path.insert(0, str(code_dir))
    try:
        if use_cpu_kernels:
            shim = types.ModuleType("kernel")
            for fn in ("act_quant", "fp4_act_quant", "fp8_gemm", "fp4_gemm", "sparse_attn", "hc_split_sinkhorn"):
                setattr(shim, fn, getattr(cpu_kernels, fn))
            sys.modules["kernel"] = shim
        spec = importlib.util.spec_from_file_location("model", code_dir / "model.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules["model"] = module
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.path.remove(str(code_dir))
        for n in names:
            sys.modules.pop(n, None)
        sys.modules.update(saved)


def convert_checkpoint(code_dir: Path, hf_ckpt: Path, out_dir: Path, expert_dtype: str | None) -> Path:
    """Run the reference convert.py with model-parallel 1; returns the converted shard path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "model0-mp1.safetensors"
    if target.exists():
        return target
    spec = importlib.util.spec_from_file_location("ds_ref_convert", code_dir / "convert.py")
    conv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conv)
    conv.main(str(hf_ckpt), str(out_dir), 1, expert_dtype)
    return target


class OfficialReference:
    def __init__(
        self,
        code_dir: str | Path,
        hf_ckpt: str | Path,
        work_dir: str | Path,
        use_cpu_kernels: bool = True,
        text_only: bool = False,
        max_seq_len: int = 256,
        max_batch_size: int = 1,
        tokenizer=None,
    ):
        self.code_dir = Path(code_dir)
        self.hf_ckpt = Path(hf_ckpt)
        self.config = json.loads((self.hf_ckpt / "config.json").read_text())
        self.use_cpu_kernels = use_cpu_kernels
        self.args_kwargs = hf_config_to_model_args(self.config, text_only=text_only)
        self.args_kwargs.update(max_seq_len=max_seq_len, max_batch_size=max_batch_size, temperature=0)
        self.text_only = text_only
        self.work_dir = Path(work_dir)
        self.tokenizer = tokenizer

    def trace(self, input_ids: torch.Tensor) -> ReferenceTrace:
        from safetensors.torch import load_file

        ckpt = convert_checkpoint(self.code_dir, self.hf_ckpt, self.work_dir, self.args_kwargs.get("expert_dtype"))
        prev_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)  # as generate.py does
        try:
            with _isolated_import(self.code_dir, self.use_cpu_kernels) as ref:
                fields = {f.name for f in dataclasses.fields(ref.ModelArgs)}
                unknown = sorted(set(self.args_kwargs) - fields)
                args = ref.ModelArgs(**{k: v for k, v in self.args_kwargs.items() if k in fields})
                model = ref.Transformer(args, self.tokenizer) if self.tokenizer is not None else ref.Transformer(args)
                state = load_file(str(ckpt))
                if self.text_only:
                    state = {k: v for k, v in state.items() if not k.startswith(("vision.", "aligner.", "image_")) and not k.endswith("bias_vl")}
                missing, unexpected = model.load_state_dict(state, strict=False)
                if missing or unexpected:
                    raise RuntimeError(f"checkpoint/model mismatch: missing={missing[:10]} unexpected={unexpected[:10]}")
                model.eval()
                hidden: list[torch.Tensor] = []
                hooks = [blk.register_forward_hook(lambda _m, _i, out: hidden.append(out[0].detach().clone())) for blk in model.layers]
                # full-sequence logits instead of last-position only
                hooks.append(model.head.register_forward_pre_hook(lambda _m, a, kw: (a, {**kw, "full_logits": True}), with_kwargs=True))
                try:
                    out = model(input_ids)
                finally:
                    for h in hooks:
                        h.remove()
                logits = out[1] if isinstance(out, tuple) else out
        finally:
            torch.set_default_dtype(prev_dtype)
        return ReferenceTrace(
            input_ids=input_ids,
            hidden_states=hidden,
            logits=logits.float(),
            meta={
                "backend": "official",
                "code_dir": str(self.code_dir),
                "checkpoint": str(self.hf_ckpt),
                "cpu_kernels": self.use_cpu_kernels,
                "hidden_semantics": "output of each Block: full hyper-connection stream [b, s, hc_mult, dim]",
                "ignored_config_keys": unknown,
            },
        )

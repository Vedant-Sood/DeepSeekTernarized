"""Export a QAT-trained model to Boilerplate Mode D (and from there to GGUF TQ1_0/TQ2_0).

Every BitLinear becomes one absmean ternary term (exactly what its forward used at level 1);
everything else (embeddings, LM head, norms, routers) is written in the master dtype as Mode A.
The result loads in the standard-decoder engine like any other export.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch import nn

from ..format.export import export
from ..format.policy import Policy, Rule
from .bitlinear import BitLinear


def bitlinear_settings(model: nn.Module) -> tuple[list[str], int, str]:
    mods = [(n, m) for n, m in model.named_modules() if isinstance(m, BitLinear)]
    if not mods:
        raise ValueError("model has no BitLinear layers")
    groups = {m.group for _, m in mods}
    scales = {m.scale_dtype for _, m in mods}
    if len(groups) != 1 or len(scales) != 1:
        raise ValueError("all BitLinear layers must share group size and scale dtype")
    scale = {torch.float16: "fp16", torch.bfloat16: "bf16"}[scales.pop()]
    return [n for n, _ in mods], groups.pop(), scale


def _source_dtypes(source_dir: Path) -> dict[str, torch.dtype]:
    from safetensors import safe_open

    out = {}
    for f in sorted(source_dir.glob("*.safetensors")):
        with safe_open(str(f), "pt") as h:
            for k in h.keys():
                out[k] = h.get_slice(k).get_dtype()
    names = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
    return {k: names[v] for k, v in out.items() if v in names}


def export_mode_d(model: nn.Module, source_dir: str | Path, out_dir: str | Path, *, packing: str = "tq1",
                  state_dict: dict[str, torch.Tensor] | None = None, log=print) -> Path:
    """Write ``model`` (or a gathered full ``state_dict`` of it) as a Mode D export.

    ``source_dir`` supplies config.json and tokenizer files. For FSDP-trained models pass the
    full state dict gathered on rank 0 (``get_model_state_dict(..., full_state_dict=True)``).
    """
    names, group, scale = bitlinear_settings(model)
    state = state_dict if state_dict is not None else model.state_dict()
    tensors = {k: v.detach().cpu().contiguous() for k, v in state.items() if not k.endswith("_extra_state")}
    if getattr(model.config, "tie_word_embeddings", False):
        tensors.pop("lm_head.weight", None)  # shared with the embedding; the engine reads it from there
    source_dir = Path(source_dir)
    # Tensors that stay Mode A (embeddings, head, norms, routers) go back to the source's dtype:
    # training them in an FP32 master copy must not double the exported file. BitLinear weights
    # keep the master precision, since the ternary fit is taken from them.
    ternary = {f"{n}.weight" for n in names}
    source_dtypes = _source_dtypes(source_dir)
    for k, v in tensors.items():
        want = source_dtypes.get(k)
        if k not in ternary and want is not None and v.is_floating_point() and v.dtype != want:
            tensors[k] = v.to(want)
    with tempfile.TemporaryDirectory(prefix="mode-d-") as tmp:
        tmp = Path(tmp)
        save_file(tensors, str(tmp / "model.safetensors"))
        shutil.copy(source_dir / "config.json", tmp / "config.json")
        for f in ("tokenizer.json", "tokenizer_config.json"):
            if (source_dir / f).exists():
                shutil.copy(source_dir / f, tmp / f)
        pattern = "^(" + "|".join(n.replace(".", r"\.") for n in names) + ")$"
        policy = Policy("mode-d-qat", [Rule(match=pattern, mode="D", group=group, packing=packing,
                                            scale_dtype=scale, act="int8")])
        out = export(tmp, out_dir, policy, log=log)
    meta = json.loads((out / "manifest.json").read_text())
    meta["source"] = str(source_dir)
    (out / "manifest.json").write_text(json.dumps(meta, indent=1))
    return out

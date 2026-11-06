"""Write Boilerplate single-term ternary weights as llama.cpp ``TQ1_0`` / ``TQ2_0`` GGUF tensors.

Goal: run llama.cpp's ternary kernels on *exactly* the weights our engine runs, so speed and quality
comparisons are on identical numbers. ggml's TQ formats hold blocks of 256 trits with one FP16
scale per block, so a tensor converts exactly only if it is Mode C with k = 1, ``group: 256``,
``scale_dtype: fp16`` and its input dimension a multiple of 256. Anything else is reported and not
converted (llama.cpp itself would fall back to another type there too).

Two steps, because only llama.cpp's converter writes the tokenizer and hyperparameter metadata:

1. :func:`ternary_gguf_tensors` turns an export directory into {gguf tensor name: FP32 values}
   in llama.cpp's naming and layout (Llama Q/K rows permuted as ``convert_hf_to_gguf`` does,
   OLMoE experts stacked into ``*_exps`` tensors).
2. :func:`merge_into_base` copies a BF16/F32 GGUF made by llama.cpp's ``convert_hf_to_gguf.py``
   from the same source checkpoint and swaps those tensors for ``TQ1_0``/``TQ2_0`` data, checking
   that every swapped block dequantizes back to our values bit for bit.

Requires the ``gguf`` Python package (``pip install gguf``).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import torch

from ..engine.weights import WeightStore
from ..ternary.pack import unpack

QK = 256


def llama_permute(w: torch.Tensor, n_head: int, n_head_kv: int | None) -> torch.Tensor:
    """llama.cpp ``LlamaModel.permute``: Q/K rows reordered for its RoPE layout."""
    if n_head_kv is not None and n_head != n_head_kv:
        n_head = n_head_kv
    return w.reshape(n_head, 2, w.shape[0] // n_head // 2, *w.shape[1:]).swapaxes(1, 2).reshape(w.shape)


def _single_term(store: WeightStore, base: str, meta: dict) -> tuple[torch.Tensor | None, str | None]:
    if meta.get("mode") not in ("C", "D"):
        return None, "not ternary"
    K = meta["shape"][1]
    if meta["terms"] != 1:
        return None, f"k = {meta['terms']} (TQ holds one term)"
    if meta["group"] != QK or meta["scale_dtype"] != "fp16":
        return None, f"group {meta['group']} / {meta['scale_dtype']} scales (TQ needs group 256, fp16)"
    if K % QK:
        return None, f"input dim {K} not a multiple of 256"
    trits = unpack(store.get(f"{base}.tq0.packed"), meta["packing"], K).float()
    scale = store.get(f"{base}.tq0.scale").float()
    return trits * scale.repeat_interleave(QK, -1)[:, :K], None


def ternary_gguf_tensors(export_dir: str | Path) -> tuple[str, dict[str, torch.Tensor], dict[str, str]]:
    """(gguf architecture, {gguf name: FP32 values}, {hf base: reason skipped})."""
    import gguf

    export_dir = Path(export_dir)
    manifest = json.loads((export_dir / "manifest.json").read_text())
    config = manifest["architecture"]
    model_type = config.get("model_type")
    arch = {"llama": gguf.MODEL_ARCH.LLAMA, "olmoe": gguf.MODEL_ARCH.OLMOE}.get(model_type)
    if arch is None:
        raise ValueError(f"GGUF export supports llama and olmoe, not {model_type!r}")
    names = gguf.get_tensor_name_map(arch, config["num_hidden_layers"])
    store = WeightStore(export_dir)
    out: dict[str, torch.Tensor] = {}
    skipped: dict[str, str] = {}
    experts: dict[tuple[int, str], dict[int, torch.Tensor]] = {}
    for base, meta in manifest["tensors"].items():
        if meta.get("mode") not in ("C", "D"):
            continue
        values, reason = _single_term(store, base, meta)
        if reason:
            skipped[base] = reason
            continue
        m = re.fullmatch(r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj", base)
        if m:
            experts.setdefault((int(m.group(1)), m.group(3)), {})[int(m.group(2))] = values
            continue
        if model_type == "llama" and base.endswith(("self_attn.q_proj", "self_attn.k_proj")):
            kv = config.get("num_key_value_heads", config["num_attention_heads"]) if base.endswith("k_proj") else config["num_attention_heads"]
            values = llama_permute(values, config["num_attention_heads"], kv)
        name = names.get_name(base + ".weight", try_suffixes=(".weight",))
        if name is None:
            raise ValueError(f"no llama.cpp name for {base}")
        out[name] = values
    n_experts = config.get("num_experts", 0)
    for (layer, proj), per_expert in sorted(experts.items()):
        if len(per_expert) != n_experts:
            for e in range(n_experts):
                skipped.setdefault(f"model.layers.{layer}.mlp.experts.{e}.{proj}_proj",
                                   "some experts of this layer are not convertible (GGUF stacks all experts)")
            continue
        merged = f"model.layers.{layer}.mlp.experts.{proj}_proj.weight"
        name = names.get_name(merged, try_suffixes=(".weight",))
        out[name] = torch.stack([per_expert[e] for e in range(n_experts)])
    return arch, out, skipped


def quantize_exact(values: torch.Tensor, qtype) -> np.ndarray:
    """Quantize to TQ1_0/TQ2_0 and verify the blocks decode to exactly ``values``."""
    from gguf import quants

    arr = values.float().numpy()
    q = quants.quantize(arr, qtype)
    back = quants.dequantize(q, qtype).reshape(arr.shape)
    if not np.array_equal(back, arr):
        bad = int((back != arr).sum())
        raise ValueError(f"{bad} values do not survive the TQ round trip exactly")
    return q


def merge_into_base(base_gguf: str | Path, export_dir: str | Path, out_path: str | Path,
                    packing: str = "tq1") -> dict:
    """Copy ``base_gguf`` (llama.cpp conversion of the same source) with our ternary tensors swapped in."""
    import gguf

    qtype = {"tq1": gguf.GGMLQuantizationType.TQ1_0, "tq2": gguf.GGMLQuantizationType.TQ2_0}[packing]
    arch, values, skipped = ternary_gguf_tensors(export_dir)
    reader = gguf.GGUFReader(str(base_gguf))
    arch_field = reader.fields[gguf.Keys.General.ARCHITECTURE]
    base_arch = arch_field.contents()
    if base_arch != gguf.MODEL_ARCH_NAMES[arch]:
        raise ValueError(f"base GGUF is {base_arch!r}, export is {gguf.MODEL_ARCH_NAMES[arch]!r}")
    present = {t.name for t in reader.tensors}
    missing = sorted(set(values) - present)
    if missing:
        raise ValueError(f"tensors absent from the base GGUF: {missing[:5]}")
    quantized = {}
    for t in reader.tensors:
        if t.name in values:
            if tuple(int(x) for x in reversed(t.shape.tolist())) != tuple(values[t.name].shape):
                raise ValueError(f"{t.name}: shape {tuple(values[t.name].shape)} differs from the base GGUF")
            quantized[t.name] = quantize_exact(values[t.name], qtype)
    writer = gguf.GGUFWriter(str(out_path), base_arch, endianess=reader.endianess)
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), val_type, sub_type=sub_type)
    writer.add_string("boilerplate.ternary_source", str(Path(export_dir).resolve().name))
    for t in reader.tensors:
        if t.name in quantized:
            q = quantized[t.name]
            writer.add_tensor_info(t.name, q.shape, q.dtype, q.nbytes, qtype)
        else:
            writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for t in reader.tensors:
        writer.write_tensor_data(quantized.get(t.name, t.data), tensor_endianess=reader.endianess)
    writer.close()
    return {"converted": sorted(quantized), "skipped": skipped, "qtype": qtype.name}


def write_ternary_only(export_dir: str | Path, out_path: str | Path, packing: str = "tq1") -> dict:
    """A GGUF holding only the converted tensors (for inspection, checksums and tests)."""
    import gguf

    qtype = {"tq1": gguf.GGMLQuantizationType.TQ1_0, "tq2": gguf.GGMLQuantizationType.TQ2_0}[packing]
    arch, values, skipped = ternary_gguf_tensors(export_dir)
    writer = gguf.GGUFWriter(str(out_path), gguf.MODEL_ARCH_NAMES[arch])
    for name, v in sorted(values.items()):
        writer.add_tensor(name, quantize_exact(v, qtype), raw_dtype=qtype)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return {"converted": sorted(values), "skipped": skipped, "qtype": qtype.name}

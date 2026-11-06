"""Export a checkpoint to the Boilerplate format under a precision policy.

Output directory:

* ``config.json`` and tokenizer files, copied from the source.
* ``model-XXXXX-of-YYYYY.safetensors`` shards. Tensor naming per mode, for a weight with base name
  ``X`` (its name without ``.weight``):
  - Mode A: ``X.weight`` (and ``X.scale``) exactly as in the source.
  - Mode B: for each source tensor ``T`` of ``X``: ``T.hstream`` (uint8 Huffman payload),
    ``T.hoffsets`` (int64 block bit offsets), ``T.hlens`` (uint8[256] code lengths) and, for
    2-byte dtypes, ``T.hraw`` (the uncoded low bytes). See ``entropy.py`` for the stream layout.
  - Mode C: ``X.tq{i}.packed`` (uint8, TQ2/TQ1 rows, see ``ternary/pack.py``) and ``X.tq{i}.scale``
    ([N, ceil(K / group)], FP16/BF16) for each term i.
  - Mode D: stored exactly like Mode C with one term, quantized with the absmean rule that QAT
    trains against (``boilerplate.qat.export`` writes it from a trained model).
  Supported one- and two-byte tensors outside a weight/scale pair can also use Mode B;
  other non-weight tensors stay in their original format.
* ``manifest.json``: format version, architecture (the source config), policy, and per weight:
  mode, logical shape, source dtype, packing, terms, group size, scale dtype, activation mode, the
  stored tensor names with SHA-256 checksums, stored bytes; plus a size summary with effective
  bits per weight over the whole model (scales and high-precision tensors included).
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file

from ..checkpoint.dequant import dequant_fp4, dequant_fp8_block, is_packed_fp4
from ..engine.weights import WeightStore
from ..ternary.fit import absmean_ternary, fit_gptq, fit_residual
from ..ternary.pack import pack
from . import entropy
from .policy import MAX_RATIO, Policy

FORMAT_VERSION = 1
FP4_STORAGE = (torch.int8, torch.uint8, torch.float4_e2m1fn_x2)
SCALE_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}


def logical_numel(t: torch.Tensor, is_fp4: bool) -> int:
    return t.numel() * 2 if is_fp4 else t.numel()


def bases_of(names: list[str]) -> dict[str, list[str]]:
    """base name -> its tensors, for every ``X.weight`` (with its ``X.scale`` if present)."""
    out = {}
    s = set(names)
    for n in names:
        if n.endswith(".weight"):
            b = n[: -len(".weight")]
            out[b] = [n] + ([f"{b}.scale"] if f"{b}.scale" in s else [])
    return out


def dequant_source(w: torch.Tensor, scale: torch.Tensor | None) -> torch.Tensor:
    """FP32 values of a source weight, whatever its storage."""
    if scale is None:
        return w.float()
    if w.dtype in FP4_STORAGE and is_packed_fp4(tuple(w.shape), tuple(scale.shape)):
        return dequant_fp4(w, scale, torch.float32)
    return dequant_fp8_block(w, scale, torch.float32)


def sha256(t: torch.Tensor) -> str:
    return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


class _ShardWriter:
    """Accumulates tensors into shards. ``on_flush(files)`` runs after each shard is on disk, so
    the export journal can record what is safely written (resume after interruption)."""

    def __init__(self, out: Path, shard_bytes: int, files: list | None = None, on_flush=None):
        self.out, self.limit = out, shard_bytes
        self.cur: dict[str, torch.Tensor] = {}
        self.size = 0
        self.files: list[tuple[str, list[str]]] = [tuple(x) for x in (files or [])]
        self.k = len(self.files)
        self.on_flush = on_flush

    def add(self, name: str, t: torch.Tensor):
        nb = t.numel() * t.element_size()
        if self.cur and self.size + nb > self.limit:
            self.flush()
        self.cur[name] = t.contiguous()
        self.size += nb

    def flush(self):
        if not self.cur:
            return
        self.k += 1
        fname = f"shard-{self.k:05d}.safetensors"
        save_file(self.cur, str(self.out / fname))
        self.files.append((fname, list(self.cur)))
        self.cur, self.size = {}, 0
        if self.on_flush is not None:
            self.on_flush(self.files)

    def finish(self) -> dict[str, str]:
        self.flush()
        n = len(self.files)
        wm = {}
        for i, (f, names) in enumerate(self.files):
            final = f"model-{i + 1:05d}-of-{n:05d}.safetensors"
            (self.out / f).rename(self.out / final)
            wm.update({x: final for x in names})
        return wm


def export(
    source: str | Path,
    out_dir: str | Path,
    policy: Policy,
    hessians: dict[str, torch.Tensor] | None = None,
    shard_bytes: int = 2 << 30,
    log=print,
    resume: bool = True,
) -> Path:
    """Stream ``source`` tensor by tensor into ``out_dir`` under ``policy``.

    Restartable: ``out_dir/.export_journal.json`` lists every shard already written and the
    manifest entries of the weights inside it. A rerun with the same source and policy skips
    those weights (their shards are kept, checked by size) and continues; another policy or
    source starts over. Peak memory is one source tensor plus one output shard.
    """
    src, out = Path(source), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ws = WeightStore(src)
    names = ws.names()
    bases = bases_of(names)
    in_base = {t for ts in bases.values() for t in ts}
    journal_path = out / ".export_journal.json"
    identity = {"source": str(src.resolve()), "policy": policy.to_dict(), "shard_bytes": shard_bytes,
                "hessians": sorted(hessians) if hessians else []}
    tensors_meta: dict[str, dict] = {}
    pending: dict[str, dict] = {}
    files: list = []
    if resume and journal_path.exists():
        j = json.loads(journal_path.read_text())
        if j["identity"] == identity and all((out / f).exists() and (out / f).stat().st_size == s
                                             for f, s in j["sizes"].items()):
            files, tensors_meta = j["files"], j["tensors"]
            log(f"resuming export: {len(tensors_meta)} entries in {len(files)} shards already written")
        else:
            log("export journal is for another source/policy or its shards changed: starting over")

    def on_flush(written):
        flushed = {n for _, ns in written for n in ns}
        for base in [b for b, m in pending.items() if set(m["stored"]) <= flushed]:
            tensors_meta[base] = pending.pop(base)
        tmp = journal_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"identity": identity, "files": written, "tensors": tensors_meta,
                                   "sizes": {f: (out / f).stat().st_size for f, _ in written}}))
        tmp.replace(journal_path)

    writer = _ShardWriter(out, shard_bytes, files, on_flush)

    def record(base, meta, stored: dict[str, torch.Tensor], params: int):
        nb = 0
        meta["stored"] = {}
        for n, t in stored.items():
            writer.add(n, t)
            b = t.numel() * t.element_size()
            nb += b
            meta["stored"][n] = {"dtype": str(t.dtype).replace("torch.", ""), "shape": list(t.shape), "sha256": sha256(t)}
        meta["bytes"] = nb
        meta["params"] = params
        pending[base] = meta

    def encode_lossless(tnames):
        stored, parts_meta = {}, {}
        for n in tnames:
            t = ws.get(n)
            if t.element_size() > 2:
                stored[n] = t
                continue
            p = entropy.encode(t)
            parts_meta[n] = {"dtype": str(t.dtype).replace("torch.", ""), "shape": list(t.shape),
                             "symbols": entropy.split_bytes(t)[0].numel()}
            stored[f"{n}.hstream"] = p["stream"]
            stored[f"{n}.hoffsets"] = p["offsets"]
            stored[f"{n}.hlens"] = p["lens"]
            if "raw" in p:
                stored[f"{n}.hraw"] = p["raw"]
        return stored, parts_meta

    done = 0
    for base, tnames in sorted(bases.items()):
        if base in tensors_meta:  # written before an interruption
            done += 1
            continue
        rule = policy.rule_for(base)
        w = ws.get(tnames[0])
        scale = ws.get(tnames[1]) if len(tnames) > 1 else None
        fp4 = scale is not None and w.dtype in FP4_STORAGE and is_packed_fp4(tuple(w.shape), tuple(scale.shape))
        params = logical_numel(w, fp4)
        meta = {"mode": rule.mode, "shape": [w.size(0), w.size(1) * (2 if fp4 else 1)] if w.ndim == 2 else list(w.shape),
                "source_dtype": "fp4_e2m1" if fp4 else str(w.dtype).replace("torch.", "")}
        if rule.mode in ("C", "D") and w.ndim == 2:
            src_bits = 8 * sum(ws.get(n).numel() * ws.get(n).element_size() for n in tnames) / params
            effective_bits = rule.effective_bits(meta["shape"][1])
            if effective_bits >= MAX_RATIO * src_bits and not rule.allow_larger:
                raise ValueError(
                    f"{base}: Mode C {rule.packing} k={rule.k} group={rule.group} is {effective_bits:.3f} bits/weight, "
                    f"not below {MAX_RATIO} x the source's {src_bits:.3f}; lower k, use tq1, or set allow_larger"
                )
            meta["source_bits"] = src_bits
            W = dequant_source(w, scale)
            gptq = rule.gptq
            if gptq and hessians is None:
                raise ValueError(f"{base}: gptq requested but no calibration Hessians were passed")
            if gptq and base not in hessians:
                # calibration routed too few tokens here (rare MoE experts): plain fit, recorded
                gptq = False
                meta["gptq_fallback"] = "too few calibration tokens"
            if gptq:
                terms = fit_gptq(W, hessians[base], rule.k, rule.group, scale_dtype=SCALE_DTYPES[rule.scale_dtype],
                                 quantizer=rule.quantizer)
            elif rule.quantizer == "absmean":
                terms = [absmean_ternary(W, rule.group, SCALE_DTYPES[rule.scale_dtype])]
            else:
                terms = fit_residual(W, rule.k, rule.group, SCALE_DTYPES[rule.scale_dtype])
            stored = {}
            for i, t in enumerate(terms):
                stored[f"{base}.tq{i}.packed"] = pack(t.trits, rule.packing)
                stored[f"{base}.tq{i}.scale"] = t.scale
            meta.update(packing=rule.packing, terms=rule.k, group=rule.group, scale_dtype=rule.scale_dtype,
                        act=rule.act, gptq=gptq, fit=rule.fit, quantizer=rule.quantizer)
            record(base, meta, stored, params)
        elif rule.mode == "B":
            stored, parts_meta = encode_lossless(tnames)
            meta.update(coded=parts_meta, maxlen=entropy.MAXLEN, block=entropy.BLOCK)
            record(base, meta, stored, params)
        else:
            meta["mode"] = "A"
            record(base, meta, {n: ws.get(n) for n in tnames}, params)
        done += 1
        if done % 200 == 0:
            log(f"  {done}/{len(bases)} weights")
    # Non-weight tensors can use Mode B too; wider dtypes and Mode C selections stay original.
    for n in names:
        if n not in in_base and n not in tensors_meta:
            t = ws.get(n)
            meta = {"mode": "A", "shape": list(t.shape), "source_dtype": str(t.dtype).replace("torch.", "")}
            if policy.rule_for(n).mode == "B" and t.element_size() <= 2:
                stored, parts_meta = encode_lossless([n])
                meta.update(mode="B", coded=parts_meta, maxlen=entropy.MAXLEN, block=entropy.BLOCK)
            else:
                stored = {n: t}
            record(n, meta, stored, t.numel())

    weight_map = writer.finish()
    tensors_meta.update(pending)
    pending.clear()
    tensors_meta = dict(sorted(tensors_meta.items()))
    total_params = sum(m["params"] for m in tensors_meta.values())
    total_bytes = sum(m["bytes"] for m in tensors_meta.values())
    by_mode: dict[str, list[int]] = {}
    for m in tensors_meta.values():
        acc = by_mode.setdefault(m["mode"], [0, 0])
        acc[0] += m["params"]
        acc[1] += m["bytes"]
    for meta in tensors_meta.values():
        for n in meta["stored"]:
            meta["stored"][n]["file"] = weight_map[n]
    config = json.loads((src / "config.json").read_text())
    manifest = {
        "format": "boilerplate",
        "version": FORMAT_VERSION,
        "source": str(src),
        "architecture": config,
        "policy": policy.to_dict(),
        "summary": {
            "params": total_params,
            "bytes": total_bytes,
            "bits_per_weight": 8 * total_bytes / max(total_params, 1),
            "by_mode": {m: {"params": p, "bytes": b, "bits_per_weight": 8 * b / max(p, 1)} for m, (p, b) in by_mode.items()},
        },
        "tensors": tensors_meta,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    (out / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": total_bytes}, "weight_map": weight_map}))
    journal_path.unlink(missing_ok=True)
    shutil.copy(src / "config.json", out / "config.json")
    for f in ("tokenizer.json", "tokenizer_config.json"):
        if (src / f).exists():
            shutil.copy(src / f, out / f)
    return out

"""Cast a Hugging Face safetensors checkpoint to another float dtype, shard by shard.

Used for OLMoE-1B-7B-0125, whose published weights are F32 while the M4 plan assumes a BF16
source (decision 2026-11-05: convert to BF16 and use that as the reference; the code accepts
either). The cast is lossy (round-to-nearest-even) and is recorded in ``cast.json`` with the
SHA-256 of every input and output shard so results can name exactly which source they used.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def cast_checkpoint(src: str | Path, dst: str | Path, dtype: str = "bf16", log=print) -> Path:
    src, dst = Path(src), Path(dst)
    target = DTYPES[dtype]
    dst.mkdir(parents=True, exist_ok=True)
    shards = sorted(src.glob("*.safetensors"))
    if not shards:
        raise ValueError(f"no safetensors shards in {src}")
    record = {"source": str(src), "dtype": dtype, "rounding": "round-to-nearest-even (torch .to)", "shards": {}}
    for shard in shards:
        out = dst / shard.name
        tensors, kept = {}, 0
        with safe_open(str(shard), framework="pt") as f:
            meta = f.metadata()
            for name in f.keys():
                t = f.get_tensor(name)
                if t.is_floating_point() and t.dtype != target:
                    t = t.to(target)
                else:
                    kept += 1
                tensors[name] = t.contiguous()
        save_file(tensors, str(out), metadata=meta)
        record["shards"][shard.name] = {"source_sha256": _sha(shard), "sha256": _sha(out),
                                        "tensors": len(tensors), "unchanged": kept}
        log(f"{shard.name}: {len(tensors)} tensors")
    for f in src.iterdir():
        if f.is_file() and f.suffix != ".safetensors" and not (dst / f.name).exists():
            shutil.copy(f, dst / f.name)
    index = dst / "model.safetensors.index.json"
    if index.exists():  # sizes changed; the weight map did not
        data = json.loads(index.read_text())
        data.setdefault("metadata", {})["total_size"] = sum((dst / s).stat().st_size for s in record["shards"])
        index.write_text(json.dumps(data, indent=2))
    cfg = dst / "config.json"
    if cfg.exists():
        c = json.loads(cfg.read_text())
        c["torch_dtype"] = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}[dtype]
        c.pop("dtype", None)
        cfg.write_text(json.dumps(c, indent=2))
    (dst / "cast.json").write_text(json.dumps(record, indent=2))
    return dst

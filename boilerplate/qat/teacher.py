"""Offline teacher logits for distillation on small GPUs.

Keeping the BF16 teacher resident doubles memory; on an 8 GB card that does not fit. Instead the
teacher runs once over the training windows and its top-k log probabilities (plus the aggregate
tail, see ``distill.teacher_topk``) are stored per window. The trainer reads them by window id.

Layout: ``DIR/meta.json`` and shards ``DIR/teacher-XXXXX.safetensors`` with tensors per window i:
``{i}.indices`` [L, k] (uint16 when the vocabulary fits, else int32), ``{i}.log_probs`` [L, k]
FP16, ``{i}.tail`` [L] FP32. Storage is about L * k * 4 bytes per window (k = 32, L = 2048:
256 KiB), i.e. ~128 KB per 1 k tokens: plan disk accordingly (1 B tokens ≈ 128 GB at k = 32).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .data import TokenWindows
from .distill import teacher_topk


def precompute(model, windows: TokenWindows, out_dir: str | Path, window_ids: list[int], *, k: int = 32,
               temperature: float = 1.0, batch: int = 1, per_shard: int = 256, device: str = "cpu",
               log=print) -> Path:
    """Run ``model`` (an HF causal LM, the unquantized teacher) and store top-k per window.
    Resumable: shards already written are skipped."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    vocab = model.config.vocab_size
    meta = {"k": k, "temperature": temperature, "seq_len": windows.seq_len, "vocab_size": vocab,
            "token_file_sha256": windows.meta.get("sha256"), "per_shard": per_shard, "shards": {}}
    mpath = out / "meta.json"
    if mpath.exists():
        old = json.loads(mpath.read_text())
        if {x: old[x] for x in ("k", "temperature", "seq_len", "token_file_sha256")} != {x: meta[x] for x in ("k", "temperature", "seq_len", "token_file_sha256")}:
            raise ValueError("existing teacher directory was built with other settings")
        meta["shards"] = old["shards"]
    idx_dtype = torch.uint16 if vocab <= 1 << 16 else torch.int32
    model.eval().to(device)
    for s in range(0, len(window_ids), per_shard):
        name = f"teacher-{s // per_shard:05d}.safetensors"
        ids = window_ids[s:s + per_shard]
        if name in meta["shards"]:
            continue
        tensors = {}
        for b in range(0, len(ids), batch):
            chunk = ids[b:b + batch]
            x = torch.stack([windows.window(i)[:-1] for i in chunk]).to(device)
            with torch.inference_mode():
                logits = model(x).logits.float()
            t = teacher_topk(logits, k, temperature)
            for j, i in enumerate(chunk):
                tensors[f"{i}.indices"] = t["indices"][j].to(torch.int32).to(idx_dtype).cpu().contiguous()
                tensors[f"{i}.log_probs"] = t["log_probs"][j].half().cpu().contiguous()
                tensors[f"{i}.tail"] = t["tail_log_prob"][j].float().cpu().contiguous()
        save_file(tensors, str(out / name))
        meta["shards"][name] = ids
        mpath.write_text(json.dumps(meta))
        log(f"teacher shard {name}: {len(ids)} windows")
    return out


class TeacherStore:
    """Reads precomputed teacher top-k by window id, in the dict form ``topk_teacher_kl`` takes."""

    def __init__(self, directory: str | Path):
        self.dir = Path(directory)
        self.meta = json.loads((self.dir / "meta.json").read_text())
        self.where = {i: name for name, ids in self.meta["shards"].items() for i in ids}
        self._open: dict[str, object] = {}

    def _shard(self, name: str):
        # memory-mapped handles: reading one window touches only its bytes
        if name not in self._open:
            self._open[name] = safe_open(str(self.dir / name), framework="pt")
        return self._open[name]

    def get(self, window_ids: list[int]) -> dict[str, torch.Tensor]:
        missing = [i for i in window_ids if i not in self.where]
        if missing:
            raise KeyError(f"no teacher outputs for windows {missing[:5]}")
        rows = [self._shard(self.where[i]) for i in window_ids]
        return {
            "indices": torch.stack([r.get_tensor(f"{i}.indices").to(torch.int64) for r, i in zip(rows, window_ids)]),
            "log_probs": torch.stack([r.get_tensor(f"{i}.log_probs").float() for r, i in zip(rows, window_ids)]),
            "tail_log_prob": torch.stack([r.get_tensor(f"{i}.tail") for r, i in zip(rows, window_ids)]),
            "temperature": torch.tensor(self.meta["temperature"]),
        }

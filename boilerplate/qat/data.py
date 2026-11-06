"""Token data for QAT: flat token files, fixed-length windows, a resumable deterministic sampler.

On disk a corpus is one ``.bin`` file of little-endian token ids (uint16 when the vocabulary fits,
else uint32) plus a ``.json`` sidecar with dtype, token count, tokenizer SHA-256 and source. It is
memory-mapped, so a multi-billion-token file costs no RAM.

Training samples are windows ``[i*L, (i+1)*L + 1)`` (inputs plus the shifted labels). The window
order is a seeded permutation per epoch; each rank takes every ``world``-th window, so the
sampler state is just ``(epoch, position)`` and resume is exact. Offline teacher top-k files
(``teacher.py``) are indexed by the same window ids.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def write_tokens(ids, out: str | Path, *, vocab_size: int, tokenizer_sha256: str = "", source: str = "") -> Path:
    """Write a token file and its sidecar; ``ids`` is any iterable of int lists or a flat array."""
    out = Path(out)
    dtype = np.uint16 if vocab_size <= 1 << 16 else np.uint32
    if isinstance(ids, np.ndarray):
        arr = ids.astype(dtype)
    else:
        arr = np.fromiter((t for chunk in ids for t in chunk), dtype=dtype)
    if arr.size and int(arr.max()) >= vocab_size:
        raise ValueError("token id outside the vocabulary")
    out.parent.mkdir(parents=True, exist_ok=True)
    arr.tofile(out)
    meta = {"dtype": np.dtype(dtype).name, "tokens": int(arr.size), "vocab_size": vocab_size,
            "tokenizer_sha256": tokenizer_sha256, "source": source,
            "sha256": hashlib.sha256(arr.tobytes()).hexdigest()}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return out


def tokenize_text_file(text_path: str | Path, tokenizer_path: str | Path, out: str | Path,
                       *, chunk_chars: int = 1 << 22) -> Path:
    """Tokenize a UTF-8 text file (no BOS, no special-token parsing) into a token file."""
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(tokenizer_path))
    text = Path(text_path).read_text(encoding="utf-8")
    # Chunk on newlines so a long corpus does not need one giant encode call.
    chunks, start = [], 0
    while start < len(text):
        end = min(len(text), start + chunk_chars)
        if end < len(text):
            nl = text.rfind("\n", start, end)
            end = nl + 1 if nl > start else end
        chunks.append(tok.encode(text[start:end], add_special_tokens=False).ids)
        start = end
    sha = hashlib.sha256(Path(tokenizer_path).read_bytes()).hexdigest()
    return write_tokens(chunks, out, vocab_size=tok.get_vocab_size(), tokenizer_sha256=sha, source=str(text_path))


class TokenWindows:
    """Fixed-length training windows over a memory-mapped token file."""

    def __init__(self, path: str | Path, seq_len: int):
        path = Path(path)
        self.meta = json.loads(path.with_suffix(".json").read_text())
        self.tokens = np.memmap(path, dtype=self.meta["dtype"], mode="r", shape=(self.meta["tokens"],))
        self.seq_len = seq_len
        self.count = (self.meta["tokens"] - 1) // seq_len
        if self.count <= 0:
            raise ValueError(f"{path}: fewer than {seq_len + 1} tokens")

    def __len__(self) -> int:
        return self.count

    def window(self, i: int) -> torch.Tensor:
        """Tokens of window i: seq_len inputs followed by one extra token for the last label."""
        a = i * self.seq_len
        return torch.from_numpy(np.asarray(self.tokens[a:a + self.seq_len + 1], dtype=np.int64))


class Sampler:
    """Seeded per-epoch permutation, strided over ranks; ``state()`` / ``load_state()`` resume exactly."""

    def __init__(self, n_windows: int, *, seed: int, rank: int = 0, world: int = 1):
        if not 0 <= rank < world or n_windows < world:
            raise ValueError("invalid rank/world for the number of windows")
        self.n, self.seed, self.rank, self.world = n_windows, seed, rank, world
        self.epoch, self.position = 0, 0
        self._order = self._permutation()

    def _permutation(self) -> np.ndarray:
        order = np.random.default_rng([self.seed, self.epoch]).permutation(self.n)
        per_rank = self.n // self.world  # equal counts so ranks step in lockstep
        return order[self.rank:per_rank * self.world:self.world]

    def next(self, k: int) -> list[int]:
        out = []
        while len(out) < k:
            if self.position >= len(self._order):
                self.epoch += 1
                self.position = 0
                self._order = self._permutation()
            take = min(k - len(out), len(self._order) - self.position)
            out.extend(int(x) for x in self._order[self.position:self.position + take])
            self.position += take
        return out

    def state(self) -> dict:
        return {"epoch": self.epoch, "position": self.position, "seed": self.seed, "rank": self.rank, "world": self.world}

    def load_state(self, s: dict) -> None:
        if (s["seed"], s["rank"], s["world"]) != (self.seed, self.rank, self.world):
            raise ValueError("sampler state is for a different seed, rank or world size")
        self.epoch, self.position = s["epoch"], s["position"]
        self._order = self._permutation()


def batch(windows: TokenWindows, ids: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """(inputs [B, L], labels [B, L]) for the given window ids."""
    w = torch.stack([windows.window(i) for i in ids])
    return w[:, :-1].contiguous(), w[:, 1:].contiguous()

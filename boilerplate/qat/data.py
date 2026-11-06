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


class _TokenWriter:
    """Appends token ids to a ``.bin`` file in chunks, hashing as it goes (constant memory)."""

    def __init__(self, out: Path, vocab_size: int):
        self.out, self.vocab = out, vocab_size
        self.dtype = np.uint16 if vocab_size <= 1 << 16 else np.uint32
        out.parent.mkdir(parents=True, exist_ok=True)
        self.f = out.open("wb")
        self.sha = hashlib.sha256()
        self.count = 0

    def add(self, ids) -> None:
        arr = np.asarray(ids, dtype=np.int64)
        if arr.size and (int(arr.max()) >= self.vocab or int(arr.min()) < 0):
            raise ValueError("token id outside the vocabulary")
        raw = arr.astype(self.dtype).tobytes()
        self.f.write(raw)
        self.sha.update(raw)
        self.count += arr.size

    def close(self, **meta) -> Path:
        self.f.close()
        info = {"dtype": np.dtype(self.dtype).name, "tokens": self.count, "vocab_size": self.vocab,
                "sha256": self.sha.hexdigest(), **meta}
        self.out.with_suffix(".json").write_text(json.dumps(info, indent=2))
        return self.out


EOS_CANDIDATES = ("<|endoftext|>", "<|end_of_text|>", "</s>", "<|im_end|>", "<eos>")


def tokenize_text_file(text_path: str | Path, tokenizer_path: str | Path, out: str | Path,
                       *, chunk_chars: int = 1 << 22, eos_token: str | None = None) -> Path:
    """Tokenize a corpus into a token file, streaming (constant memory at any corpus size).

    A ``.txt`` file is encoded as running text (no BOS, no special-token parsing), cut at newlines
    into chunks. A ``.jsonl`` file holds one document per line (``{"text": ...}``); documents are
    encoded in batches and separated by the EOS token (``eos_token``, or the first of
    ``EOS_CANDIDATES`` the tokenizer knows), the usual pretraining layout.
    """
    from tokenizers import Tokenizer

    text_path = Path(text_path)
    tok = Tokenizer.from_file(str(tokenizer_path))
    sha = hashlib.sha256(Path(tokenizer_path).read_bytes()).hexdigest()
    w = _TokenWriter(Path(out), tok.get_vocab_size())
    meta = {"tokenizer_sha256": sha, "source": str(text_path)}
    if text_path.suffix == ".jsonl":
        names = [eos_token] if eos_token else list(EOS_CANDIDATES)
        eos = next((tok.token_to_id(n) for n in names if tok.token_to_id(n) is not None), None)
        if eos is None:
            raise ValueError(f"no EOS token among {names} in {tokenizer_path}; pass eos_token")
        docs, batch = 0, []

        def flush():
            for enc in tok.encode_batch(batch, add_special_tokens=False):
                w.add(enc.ids + [eos])
            batch.clear()

        with text_path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    batch.append(json.loads(line)["text"])
                    docs += 1
                    if len(batch) == 1024:
                        flush()
        flush()
        meta.update(documents=docs, eos_id=eos)
        return w.close(**meta)
    with text_path.open(encoding="utf-8") as f:
        carry = ""
        while True:
            block = f.read(chunk_chars)
            text = carry + block
            if not block:
                if text:
                    w.add(tok.encode(text, add_special_tokens=False).ids)
                break
            nl = text.rfind("\n")
            if nl < 0:
                carry = text
                continue
            w.add(tok.encode(text[:nl + 1], add_special_tokens=False).ids)
            carry = text[nl + 1:]
    return w.close(**meta)


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

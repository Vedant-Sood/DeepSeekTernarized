"""Engram n-gram hashing: which table rows each token position looks up.

Engram layers add an n-gram lookup to the residual stream. For every position, the
``max_ngram_size - 1`` n-grams that end there (2-gram .. max-gram) are hashed over a
*compressed* token vocabulary (tokens that normalize alike share an id), each n-gram split over
``n_heads`` heads, and each (n-gram size, head) pair owns a prime-sized bucket range of the layer's
table. Everything here is derived from config + tokenizer; nothing is hard-coded to one model.

Bit-compatible with the official ``inference/engram.py`` (checked in tests); written without its
sympy / transformers dependencies and with per-sequence history so prefill and decode of many
sequences can share one batch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


def is_prime(n: int) -> bool:
    """Deterministic Miller-Rabin for n < 3.3e24 (bases 2..41)."""
    if n < 2:
        return False
    small = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41)
    for p in small:
        if n % p == 0:
            return n == p
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in small:
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def next_unused_prime(start: int, seen: set[int]) -> int:
    c = start + 1
    while not is_prime(c) or c in seen:
        c += 1
    return c


@dataclass(frozen=True)
class EngramLayout:
    layer_ids: tuple[int, ...]
    max_ngram_size: int
    n_heads: int
    head_dim: int
    num_embeddings: tuple[int, ...]  # table rows per engram layer (from config)
    primes: tuple[tuple[tuple[int, ...], ...], ...]  # [layer][ngram size - 2][head]

    @property
    def n_hash_cols(self) -> int:
        return (self.max_ngram_size - 1) * self.n_heads

    @classmethod
    def from_config(cls, cfg: dict) -> "EngramLayout | None":
        layer_ids = tuple(cfg.get("engram_layer_ids") or ())
        if not layer_ids:
            return None
        n, heads, vocab = cfg["engram_max_ngram_size"], cfg["engram_n_heads"], cfg["engram_vocab_size"]
        seen: set[int] = set()
        primes = []
        for _ in layer_ids:
            per_ngram = []
            for _ in range(n - 1):
                cur, sizes = vocab - 1, []
                for _ in range(heads):
                    cur = next_unused_prime(cur, seen)
                    seen.add(cur)
                    sizes.append(cur)
                per_ngram.append(tuple(sizes))
            primes.append(tuple(per_ngram))
        return cls(layer_ids, n, heads, cfg["engram_head_dim"], tuple(cfg.get("engram_num_embeddings") or ()), tuple(primes))

    def table_rows(self) -> list[int]:
        """Rows each layer's table needs: the bucket ranges laid end to end."""
        return [sum(p for per in layer for p in per) for layer in self.primes]


def hash_multipliers(layer_ids, max_ngram_size: int, compressed_vocab: int) -> torch.Tensor:
    """One odd multiplier per (layer, lookback) from a per-layer seeded RNG; bounded so that
    token * multiplier fits in int64."""
    bound = max(1, (np.iinfo(np.int64).max // compressed_vocab) // 2)
    rows = []
    for lid in layer_ids:
        v = np.random.default_rng(10007 * lid).integers(0, bound, size=(max_ngram_size,), dtype=np.int64)
        rows.append(torch.tensor(v * 2 + 1))
    return torch.stack(rows)


def compressed_token_map(tokenizer) -> tuple[list[int], int]:
    """Collapse token ids whose decoded text normalizes alike (NFKC, accents stripped, lowercase,
    whitespace runs -> one space, stripped). ``tokenizer`` is a ``tokenizers.Tokenizer`` (or a
    transformers fast tokenizer, whose ``backend_tokenizer`` is used)."""
    from tokenizers import Regex, normalizers

    backend = getattr(tokenizer, "backend_tokenizer", tokenizer)
    sentinel = ""
    norm = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )
    n = len(tokenizer) if hasattr(tokenizer, "backend_tokenizer") else backend.get_vocab_size(with_added_tokens=True)
    key_to_new: dict[str, int] = {}
    lookup = [0] * n
    for tid in range(n):
        text = backend.decode([tid], skip_special_tokens=False)
        if "�" in text:
            key = backend.id_to_token(tid)
        else:
            nt = norm.normalize_str(text)
            key = nt if nt else text
        lookup[tid] = key_to_new.setdefault(key, len(key_to_new))
    return lookup, len(key_to_new)


class EngramHasher:
    """Hash ids for each position, given that position's token and its per-sequence history."""

    def __init__(self, cfg: dict, layout: EngramLayout, tokenizer):
        self.layout = layout
        token_map, vocab = compressed_token_map(tokenizer)
        expected = cfg.get("engram_compressed_vocab_size")
        if expected is not None and vocab != expected:
            raise ValueError(f"compressed engram vocab is {vocab}, config says {expected}: wrong tokenizer?")
        self.token_map = torch.tensor(token_map)
        self.pad_id = token_map[cfg.get("engram_pad_token_id", cfg.get("engram_pad_id", 2))]
        flat = [[p for per in layer for p in per] for layer in layout.primes]
        self.primes = torch.tensor(layout.primes)  # [L, n-1, H]
        self.offsets = torch.tensor(np.array([np.cumsum([0, *f[:-1]]) for f in flat]))  # [L, (n-1)*H]
        self.multipliers = hash_multipliers(layout.layer_ids, layout.max_ngram_size, vocab)  # [L, n]

    @property
    def history_len(self) -> int:
        return self.layout.max_ngram_size - 1

    def __call__(self, token_ids: torch.Tensor, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """token_ids: [s] raw ids of one sequence chunk; history: [n-1] compressed ids of the tokens
        just before it (-1 = before the start). Returns hash ids [s, L, (n-1)*H] and the new history."""
        n = self.layout.max_ngram_size
        comp = self.token_map[token_ids]
        seq = torch.cat([history, comp])  # [n-1 + s]
        s = comp.numel()
        cols = []
        blocked = torch.zeros(s, dtype=torch.bool)
        for shift in range(n):
            src = seq[n - 1 - shift : n - 1 - shift + s]
            blocked = blocked | (src < 0)
            cols.append(torch.where(blocked, self.pad_id, src))
        tokens = torch.stack(cols, -1)  # [s, n]
        products = tokens.unsqueeze(1) * self.multipliers  # [s, L, n]
        rolling, hashes = products[..., 0], []
        for i in range(1, n):
            rolling = torch.bitwise_xor(rolling, products[..., i])
            hashes.append(rolling.unsqueeze(-1) % self.primes[:, i - 1])
        out = torch.cat(hashes, -1) + self.offsets
        return out, seq[-(n - 1):].clone() if n > 1 else seq[:0]

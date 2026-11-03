"""KV cache for the V4.x sparse-attention family: a fixed ring per sequence plus paged compressed KV.

What a sequence carries between steps, per layer kind:

* **Window KV** (every layer): the last ``window_size`` post-RoPE, FP8-rounded K/V vectors (K and V
  are the same 512-d vector in this family). Bounded, so it is a per-sequence ring of
  ``window_size`` slots, slot = position % window_size, exactly like the reference.
* **Compressed KV** (KV-source layers only; the layers after a source read the source's cache): one
  vector per ``compress_ratio`` tokens, growing with the sequence. Stored in **pages**.
* **Index keys** (KV-source layers that also run an indexer): one ``index_head_dim`` vector per
  compressed position, paged alongside the compressed KV.
* **Compressor tail** (ratio > 1 sources): the projections of the last, still-incomplete group.
* **Engram history**: the last ``max_ngram_size - 1`` compressed token ids.

Paging: one page table per sequence, indexed by *token* position: logical page ``p`` covers tokens
``[p * page_size, (p + 1) * page_size)``. For a source with ratio ``r`` a physical page therefore
holds ``page_size // r`` compressed vectors, so one table serves every pool. ``page_size`` must be a
multiple of every compress ratio. Pages are allocated as a sequence grows and returned when it is
freed. The cache stores the same BF16 values the reference caches (FP8/FP4-rounded on write);
storing the narrow codes instead is a later memory optimization.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .config import EngineConfig


@dataclass
class Sequence:
    seq_id: int
    slot: int  # row in the per-sequence ring / tail buffers
    length: int = 0  # tokens already in the cache
    pages: list[int] = field(default_factory=list)
    engram_history: torch.Tensor | None = None
    tokens: list[int] = field(default_factory=list)


class KVCache:
    def __init__(
        self,
        cfg: EngineConfig,
        max_seqs: int = 8,
        max_tokens: int = 8192,
        page_size: int = 64,
        dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device = "cpu",
        engram_history_len: int = 0,
    ):
        ratios = {cfg.compress_ratios[s] for s in cfg.kv_source_layers}
        for r in ratios:
            if r and page_size % r:
                raise ValueError(f"page_size {page_size} is not a multiple of compress ratio {r}")
        self.cfg = cfg
        self.page_size = page_size
        self.num_pages = -(-max_tokens // page_size)
        self.max_seqs = max_seqs
        self.dtype = dtype
        self.engram_history_len = engram_history_len
        W, hd = cfg.window_size, cfg.head_dim
        self.window = [torch.zeros(max_seqs, W, hd, dtype=dtype, device=device) for _ in range(cfg.n_layers)]
        self.ckv: dict[int, torch.Tensor] = {}
        self.ik: dict[int, torch.Tensor] = {}
        self.tail_kv: dict[int, torch.Tensor] = {}
        self.tail_score: dict[int, torch.Tensor] = {}
        for s in cfg.kv_source_layers:
            r = cfg.compress_ratios[s]
            self.ckv[s] = torch.zeros(self.num_pages, page_size // r, hd, dtype=dtype, device=device)
            if s in cfg.index_source_layers:
                self.ik[s] = torch.zeros(self.num_pages, page_size // r, cfg.index_head_dim, dtype=dtype, device=device)
            if r > 1:
                self.tail_kv[s] = torch.zeros(max_seqs, r, hd, dtype=torch.float32, device=device)
                self.tail_score[s] = torch.full((max_seqs, r, hd), -torch.inf, dtype=torch.float32, device=device)
        self.free_pages = list(range(self.num_pages - 1, -1, -1))
        self.free_slots = list(range(max_seqs - 1, -1, -1))
        self.seqs: dict[int, Sequence] = {}
        self._next_id = 0

    # ------------------------------------------------------------------ sequence lifetime
    def new_sequence(self) -> Sequence:
        if not self.free_slots:
            raise RuntimeError(f"KV cache full: {self.max_seqs} sequences")
        seq = Sequence(self._next_id, self.free_slots.pop())
        self._next_id += 1
        if self.engram_history_len:
            seq.engram_history = torch.full((self.engram_history_len,), -1, dtype=torch.long)
        for s in self.tail_score:
            self.tail_score[s][seq.slot].fill_(-torch.inf)
            self.tail_kv[s][seq.slot].zero_()
        self.seqs[seq.seq_id] = seq
        return seq

    def free(self, seq: Sequence) -> None:
        self.free_pages.extend(reversed(seq.pages))
        seq.pages = []
        self.free_slots.append(seq.slot)
        self.seqs.pop(seq.seq_id, None)

    def reserve(self, seq: Sequence, new_length: int) -> None:
        need = -(-new_length // self.page_size)
        while len(seq.pages) < need:
            if not self.free_pages:
                raise RuntimeError("KV cache out of pages; raise max_tokens")
            seq.pages.append(self.free_pages.pop())

    @property
    def pages_in_use(self) -> int:
        return self.num_pages - len(self.free_pages)

    # ------------------------------------------------------------------ paged compressed storage
    def _addr(self, seq: Sequence, ratio: int, start: int, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        per_page = self.page_size // ratio
        j = torch.arange(start, start + n)
        table = torch.tensor(seq.pages, dtype=torch.long)
        return table[j // per_page], j % per_page

    def write_compressed(self, pool: torch.Tensor, seq: Sequence, ratio: int, start: int, values: torch.Tensor) -> None:
        """values: [n, d] for compressed indices start..start+n-1."""
        if values.size(0) == 0:
            return
        page, off = self._addr(seq, ratio, start, values.size(0))
        pool[page, off] = values.to(pool.dtype)

    def read_compressed(self, pool: torch.Tensor, seq: Sequence, ratio: int, n: int) -> torch.Tensor:
        """The first n compressed vectors of a sequence, contiguous: [n, d]."""
        if n == 0:
            return pool.new_zeros(0, pool.size(-1))
        page, off = self._addr(seq, ratio, 0, n)
        return pool[page, off]

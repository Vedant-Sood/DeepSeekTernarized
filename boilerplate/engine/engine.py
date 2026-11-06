"""Python API: load a checkpoint directory and generate, or trace a forward pass for validation.

    from boilerplate.engine import Engine
    eng = Engine.load("path/to/checkpoint")          # Mode A: weights in their stored formats
    out = eng.generate([[0, 17, 42]], max_new_tokens=16, temperature=0)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from ..reference.trace import ReferenceTrace
from .config import EngineConfig
from .kv_cache import KVCache
from .model import V4Model
from .weights import WeightStore


def load_tokenizer(path: str | Path):
    """``tokenizers.Tokenizer`` from tokenizer.json, or None when there is none."""
    p = Path(path) / "tokenizer.json"
    if not p.exists():
        return None
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(p))


@dataclass
class GenerationResult:
    tokens: list[list[int]]  # generated ids per prompt (prompt not included)
    finished: list[bool]  # stopped on EOS


def sample(logits: torch.Tensor, temperature: float, generator: torch.Generator | None = None) -> torch.Tensor:
    """Greedy at temperature 0, otherwise Gumbel-max (same distribution as the reference's sampler)."""
    if temperature == 0:
        return logits.argmax(dim=-1)
    probs = torch.softmax(logits / max(temperature, 1e-5), dim=-1, dtype=torch.float32)
    noise = torch.empty_like(probs).exponential_(1, generator=generator)
    return probs.div_(noise).argmax(dim=-1)


class Engine:
    def __init__(self, model: V4Model, tokenizer=None, max_seqs: int = 8, max_tokens: int = 16384, page_size: int = 64):
        self.model = model
        self.cfg = model.cfg
        self.tokenizer = tokenizer
        self.cache_args = dict(max_seqs=max_seqs, max_tokens=max_tokens, page_size=page_size)

    @classmethod
    def load(cls, path: str | Path, max_seq_len: int = 4096, tokenizer_path: str | Path | None = None,
             mode_b_cache_bytes: int = 256 << 20, dequant_cache_bytes: int = 2 << 30,
             expert_cache_bytes: int | None = None, **cache_args) -> "Engine":
        path = Path(path)
        config = json.loads((path / "config.json").read_text())
        if config.get("model_type") in ("llama", "olmoe"):
            from .standard import StandardEngine

            # This correctness backend keeps native or FP32 dense weights directly and uses a
            # tensor KV cache, rather than the V4 dequantization cache and paged KV allocator.
            if dequant_cache_bytes != 2 << 30:
                raise ValueError("standard decoder does not implement the V4 dequantization cache")
            device = cache_args.pop("device", None)
            if cache_args.pop("page_size", 64) != 64:
                raise ValueError("standard decoder does not implement paged KV")
            return StandardEngine.load(path, max_seq_len=max_seq_len, tokenizer_path=tokenizer_path,
                                       mode_b_cache_bytes=mode_b_cache_bytes, expert_cache_bytes=expert_cache_bytes,
                                       device=device, **cache_args)
        if cache_args.pop("ternary_backend", "auto") != "auto":
            raise ValueError("ternary_backend applies to the Llama/OLMoE decoder")
        if cache_args.pop("device", None) not in (None, "cpu"):
            raise ValueError("the V4.1 engine runs on CPU only so far; device= applies to Llama/OLMoE")
        cfg = EngineConfig.from_hf(config)
        tok = load_tokenizer(tokenizer_path or path)
        model = V4Model(cfg, WeightStore(path, mode_b_cache_bytes=mode_b_cache_bytes), tokenizer=tok,
                        max_seq_len=max_seq_len, dequant_cache_bytes=dequant_cache_bytes,
                        expert_cache_bytes=expert_cache_bytes)
        return cls(model, tok, **cache_args)

    def new_cache(self) -> KVCache:
        hist = self.model.hasher.history_len if self.model.hasher is not None else 0
        return KVCache(self.cfg, engram_history_len=hist, **self.cache_args)

    # ------------------------------------------------------------------ validation traces
    def trace(self, input_ids: torch.Tensor, decode_from: int | None = None) -> ReferenceTrace:
        """Per-layer residual streams and logits for every position, same layout as a reference
        trace. ``decode_from=k`` prefills the first k tokens and feeds the rest one at a time (each
        sequence of the batch is its own cache sequence, all decoded in one batched step)."""
        b, s = input_ids.shape
        cache = self.new_cache()
        seqs = [cache.new_sequence() for _ in range(b)]
        k = s if decode_from is None else decode_from
        cap: list = []
        logits = [self.model.forward([(q, input_ids[i, :k].tolist()) for i, q in enumerate(seqs)], cache, all_logits=True, capture=cap)]
        hidden = [[c.view(b, k, *c.shape[1:])] for c in cap]
        for t in range(k, s):
            cap = []
            lg = self.model.forward([(q, [int(input_ids[i, t])]) for i, q in enumerate(seqs)], cache, all_logits=True, capture=cap)
            logits.append(lg.view(b, 1, -1))
            for layer, c in enumerate(cap):
                hidden[layer].append(c.view(b, 1, *c.shape[1:]))
        logits[0] = logits[0].view(b, k, -1)
        return ReferenceTrace(
            input_ids=input_ids,
            hidden_states=[torch.cat(h, 1) for h in hidden],
            logits=torch.cat(logits, 1),
            meta={"backend": "boilerplate", "mode": "A", "decode_from": decode_from, "hidden_semantics": "output of each Block: full hyper-connection stream [b, s, hc_mult, dim]"},
        )

    # ------------------------------------------------------------------ generation
    def generate(
        self,
        prompts: list[list[int]],
        max_new_tokens: int = 32,
        temperature: float = 0.0,
        seed: int | None = None,
        eos_token_ids: set[int] | None = None,
        stream=None,
    ) -> GenerationResult:
        """Batched generation: one prefill step for all prompts, then one decode step per token
        for the sequences still running. ``stream(i, token_id)`` is called as tokens are produced."""
        if eos_token_ids is None:
            e = self.cfg.eos_token_id
            eos_token_ids = set(e) if isinstance(e, list) else ({e} if e is not None else set())
        gen = torch.Generator().manual_seed(seed) if seed is not None else None
        cache = self.new_cache()
        seqs = [cache.new_sequence() for _ in prompts]
        out: list[list[int]] = [[] for _ in prompts]
        done = [False] * len(prompts)
        try:
            logits = self.model.forward(list(zip(seqs, prompts)), cache, stable_sequence_hc=True)
            for step in range(max_new_tokens):
                nxt = sample(logits, temperature, gen).tolist()
                active = [i for i in range(len(prompts)) if not done[i]]
                for i, t in zip(active, nxt):
                    out[i].append(t)
                    if stream is not None:
                        stream(i, t)
                    if t in eos_token_ids:
                        done[i] = True
                active = [i for i in active if not done[i]]
                if not active or step == max_new_tokens - 1:
                    break
                logits = self.model.forward([(seqs[i], [out[i][-1]]) for i in active], cache, stable_sequence_hc=True)
        finally:
            for q in seqs:
                cache.free(q)
        return GenerationResult(out, done)

    def generate_text(self, prompt: str, **kw) -> str:
        if self.tokenizer is None:
            raise ValueError("no tokenizer.json next to the checkpoint")
        ids = self.tokenizer.encode(prompt).ids
        res = self.generate([ids], **kw)
        return self.tokenizer.decode(res.tokens[0])


"""EleutherAI lm-evaluation-harness adapter: run harness tasks in process on a Boilerplate engine.

    boilerplate lm-eval out/mode_d --tasks hellaswag,arc_easy --limit 200 --out results/lmeval.json

or, from Python, ``lm_eval.simple_evaluate(model=BoilerplateLM.load(path), tasks=[...])``. Importing
``lm_eval`` registers the model name ``boilerplate`` (``--model boilerplate --model_args
path=out/mode_d``) once this module has been imported.

The scoring is model agnostic: ``Scorer`` only needs ``engine.trace(ids).logits`` and the
tokenizer, so the same code scores the V4 engine, the standard decoder and every precision mode,
and it is tested without lm_eval installed. It follows the harness conventions:

* ``loglikelihood``: encode ``context + continuation`` as one string and split at the context's
  token count (so merges across the boundary land in the continuation); an empty context is
  prefixed with the EOS/BOS token; left truncation to the engine's context length.
* ``loglikelihood_rolling``: the whole text scored in windows of ``max_length`` targets, each with
  as much preceding context as fits (the first token is predicted from the prefix token).
* ``generate_until``: greedy or sampled generation that stops at any ``until`` string, through
  the same stop handling as ``boilerplate serve``.

Requests are scored one sequence at a time (the engines' trace path); batching is future work.
"""

from __future__ import annotations

from pathlib import Path

import torch


class Scorer:
    def __init__(self, engine, max_length: int | None = None, prefix_token_id: int | None = None):
        if engine.tokenizer is None:
            raise ValueError("lm-eval needs tokenizer.json next to the checkpoint")
        self.engine, self.tok = engine, engine.tokenizer
        self.max_length = max_length or getattr(engine, "max_seq_len", None) or engine.model.max_seq_len
        if prefix_token_id is None:
            eos = engine.cfg.eos_token_id
            prefix_token_id = eos[0] if isinstance(eos, list) else eos
            bos = getattr(engine.cfg, "bos_token_id", None)
            prefix_token_id = bos if bos is not None else prefix_token_id
        if prefix_token_id is None:
            raise ValueError("no EOS/BOS id to prefix empty contexts with; pass prefix_token_id")
        self.prefix = int(prefix_token_id)

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False).ids

    def _logprobs(self, ids: list[int]) -> torch.Tensor:
        """log-softmax over the vocabulary for every position of ``ids`` ([len(ids), vocab], float32)."""
        with torch.no_grad():
            logits = self.engine.trace(torch.tensor([ids], dtype=torch.long)).logits[0]
        return torch.log_softmax(logits.float(), dim=-1)

    def _score(self, context: list[int], target: list[int]) -> tuple[float, bool]:
        """Sum of log p(target | context) and whether every target token is the argmax
        (left-truncated to ``max_length`` positions; a target longer than that keeps its tail)."""
        full = (context + target)[-(self.max_length + 1):]
        n = min(len(target), len(full) - 1)
        lp = self._logprobs(full[:-1])[-n:]
        want = torch.tensor(full[-n:])
        return float(lp.gather(1, want[:, None]).sum()), bool((lp.argmax(-1) == want).all())

    def loglikelihood(self, context: str, continuation: str) -> tuple[float, bool]:
        if not continuation:
            raise ValueError("empty continuation")
        if not context:
            return self._score([self.prefix], self.encode(continuation))
        # trailing whitespace of the context belongs to the continuation (harness convention)
        spaces = len(context) - len(context.rstrip())
        if spaces:
            context, continuation = context[:-spaces], context[-spaces:] + continuation
        whole, ctx = self.encode(context + continuation), self.encode(context)
        target = whole[len(ctx):]
        if not target:
            raise ValueError("continuation is empty after tokenization")
        return self._score(whole[:len(ctx)] or [self.prefix], target)

    def loglikelihood_rolling(self, text: str) -> float:
        tokens = [self.prefix] + self.encode(text)
        total, start = 0.0, 1
        while start < len(tokens):
            end = min(start + self.max_length, len(tokens))
            ctx_start = max(0, end - 1 - self.max_length)
            lp = self._logprobs(tokens[ctx_start:end - 1])
            target = torch.tensor(tokens[start:end])
            total += float(lp[-len(target):].gather(1, target[:, None]).sum())
            start = end
        return total

    def generate_until(self, context: str, gen_kwargs: dict | None = None) -> str:
        from ..serve import Server

        kw = dict(gen_kwargs or {})
        until = kw.pop("until", None) or []
        until = [until] if isinstance(until, str) else list(until)
        max_toks = int(kw.pop("max_gen_toks", 256))
        do_sample = bool(kw.pop("do_sample", False))
        temperature = float(kw.pop("temperature", 0.0)) if do_sample else 0.0
        seed = kw.pop("seed", None)
        max_toks = min(max_toks, self.max_length - 1)
        ids = self.encode(context)[-(self.max_length - max_toks):] or [self.prefix]
        server = Server(self.engine, "lm-eval")
        params = {"stop": until, "max_tokens": max_toks, "n": 1, "temperature": temperature, "seed": seed}
        return server.run(ids, params)[0].text


def _lm_class():
    """The ``lm_eval.api.model.LM`` subclass (defined lazily so this module imports without lm_eval)."""
    from lm_eval.api.model import LM

    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover
        def tqdm(x, **_):
            return x

    class BoilerplateLM(LM):
        def __init__(self, path: str | None = None, engine=None, max_length: int | None = None, **engine_args):
            super().__init__()
            if engine is None:
                from ..engine import Engine

                if path is None:
                    raise ValueError("pass path= (checkpoint directory) or engine=")
                engine_args = {k: int(v) if isinstance(v, str) and v.isdigit() else v for k, v in engine_args.items()}
                engine = Engine.load(Path(path), **engine_args)
            self.scorer = Scorer(engine, max_length=int(max_length) if max_length else None)

        @classmethod
        def load(cls, path, **kw):
            return cls(path=str(path), **kw)

        def loglikelihood(self, requests, disable_tqdm: bool = False):
            return [self.scorer.loglikelihood(*r.args) for r in tqdm(requests, disable=disable_tqdm)]

        def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
            return [self.scorer.loglikelihood_rolling(r.args[0]) for r in tqdm(requests, disable=disable_tqdm)]

        def generate_until(self, requests, disable_tqdm: bool = False):
            return [self.scorer.generate_until(*r.args) for r in tqdm(requests, disable=disable_tqdm)]

    try:
        from lm_eval.api.registry import register_model

        register_model("boilerplate")(BoilerplateLM)
    except (ImportError, ValueError, AssertionError):  # already registered
        pass
    return BoilerplateLM


def __getattr__(name):
    if name == "BoilerplateLM":
        return _lm_class()
    raise AttributeError(name)


def main(a) -> int:
    import json

    import lm_eval

    torch.set_num_threads(a.threads)
    lm = _lm_class()(path=a.checkpoint, max_seq_len=a.max_seq_len, mode_b_cache_bytes=a.mode_b_cache_mib << 20)
    results = lm_eval.simple_evaluate(model=lm, tasks=a.tasks.split(","), num_fewshot=a.num_fewshot,
                                      limit=a.limit, random_seed=a.seed, numpy_random_seed=a.seed,
                                      torch_random_seed=a.seed)
    summary = {"checkpoint": a.checkpoint, "tasks": a.tasks, "limit": a.limit, "num_fewshot": a.num_fewshot,
               "results": results["results"], "versions": results.get("versions"),
               "n_samples": results.get("n-samples")}
    text = json.dumps(summary, indent=2, default=str)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text)
    print(text)
    return 0

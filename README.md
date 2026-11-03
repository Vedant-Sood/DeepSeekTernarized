# Boilerplate

A ternary-first inference engine for MoE transformer LLMs, plus a pipeline that turns
`deepseek-ai/DeepSeek-V4.1-Flash-1017` into a real ternary (~1.6–2 bits/weight) model.

Weight modes: **A** baseline (original FP8/FP4), **B** lossless entropy-coded, **C** stacked
residual ternary (k = 1–4 terms), **D** QAT-trained single ternary. Packing: TQ2 (2 bits/weight)
and TQ1 (5 trits/byte, 1.6 bits/weight).

Status: **M0** (skeleton, config parser, reference harness) and **M1** (Mode A engine, bit-exact
with the official reference on the tiny model) are done. See `docs/reports/`,
`docs/architecture/deepseek_v41_flash.md` for what the checkpoint contains and
`docs/architecture/engine.md` for the engine design.

## Quick start

```bash
pip install -e ".[hf,dev,reference]"
pytest

# Derive the architecture from a checkpoint (local dir or Hub id; headers via range requests)
boilerplate inspect deepseek-ai/DeepSeek-V4.1-Flash-1017 --json spec.json

# Tiny random-init checkpoint in the exact V4.1 on-disk format
boilerplate make-tiny ./tiny

# Reference trace with the official code (vendored in third_party/, CPU kernel shim by default)
boilerplate reference official ./tiny --code-dir third_party/deepseek_v41_inference --out traces/tiny_ref \
    --seq-len 40 --decode-from 13
# The same with the Boilerplate engine, then compare layer by layer
boilerplate trace ./tiny --out traces/tiny_modeA --seq-len 40 --decode-from 13
boilerplate compare traces/tiny_ref traces/tiny_modeA --min-cos 0.9999

# A 7-layer, 16-expert slice of the real trained weights (~6 GB by HTTP range requests)
boilerplate make-truncated deepseek-ai/DeepSeek-V4.1-Flash-1017 ./trunc7

# Generate
boilerplate generate ./trunc7 "The capital of France is" --max-new-tokens 16
```

Python API:

```python
from boilerplate.engine import Engine
eng = Engine.load("./tiny")
eng.generate([[0, 17, 42]], max_new_tokens=8, temperature=0).tokens
```

## Layout

```
boilerplate/
  checkpoint/   config/index/header reading, FP8/FP4 block formats
  config/       ModelSpec derivation, tensor roles, size report
  engine/       Mode A engine: config, mmap weights, model, paged KV cache, engram, API
  kernels/      kernel backends (torch_ref: the official kernels' numerics in PyTorch)
  reference/    traces, comparison, official + HF backends
  testing/      tiny checkpoint writer, truncated real-checkpoint slicer
third_party/    official DeepSeek V4.1 inference code (MIT), the trusted reference
configs/        source checkpoint id
docs/           architecture notes and milestone reports
tests/
```

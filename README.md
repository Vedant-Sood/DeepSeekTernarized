# Boilerplate

An inference engine for MoE transformer LLMs, being built toward a **goal** that is not yet
reached: turning `deepseek-ai/DeepSeek-V4.1-Flash-1017` into a ternary model whose routed experts
use ~1.6–2 bits/weight.

## What works today

* **Mode A only.** The engine runs the checkpoint in its shipped formats (FP8 linears, FP4 routed
  experts, BF16/FP32 small tensors). Its kernels (`boilerplate/kernels/torch_ref.py`) are a
  correctness backend: they dequantize each weight and run a floating-point GEMM. They are slow
  (0.067 tok/s decode on a 7-layer real-weight slice on 4 CPU cores).
* Mode A matches the official DeepSeek reference **bit for bit**, with both running on the same
  PyTorch kernels, on a tiny random-init model and on a 7-of-40-layer, 16-of-384-expert slice of
  the real weights without engram. That validates the engine around the kernels, not the full
  model's quality. See `docs/reports/M1.md` for the exact scope.
* No ternary code exists yet: TQ1/TQ2 packing, ternary kernels, Mode C, Mode B and QAT are planned.

## Planned weight modes

**A** baseline (original FP8/FP4), **B** lossless entropy-coded, **C** stacked residual ternary,
**D** QAT-trained single ternary. Packing: TQ2 (2 bits/weight) and TQ1 (5 trits/byte, 1.6
bits/weight), each plus an FP16 scale per 128 weights (1.725 / 2.125 bits/weight).

What ternary can save here is limited by the source: the routed experts are already FP4 (4.25
bits/weight) and ~40% of the file is FP8 engram tables. Ternary experts take the checkpoint from
~510 GB to ~334 GB (≈1.5×), and stacking two or more ternary terms on the experts saves little or
nothing over FP4. See `docs/architecture/bit_accounting.md`. Ternary weights alone are not new
(BitNet b1.58, llama.cpp `TQ1_0`/`TQ2_0`); the next experiment, a real ternary checkpoint of a
complete small model against conventional quants, is planned in `docs/plans/small_model_ternary.md`.

Milestone reports are in `docs/reports/`; `docs/architecture/deepseek_v41_flash.md` describes
what the checkpoint contains and `docs/architecture/engine.md` the engine design.

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

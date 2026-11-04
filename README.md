# Boilerplate

An inference engine for MoE transformer LLMs, being built toward a **goal** that is not yet
reached: turning `deepseek-ai/DeepSeek-V4.1-Flash-1017` into a ternary model whose routed experts
use ~1.6–2 bits/weight.

## What works today

Status: **M0** (skeleton, config parser, reference harness), **M1** (Mode A engine, bit-exact
with the official reference) and **M2** (TQ1/TQ2 packing, CPU ternary kernel, Mode C with optional
GPTQ) are implemented. **M3** has CPU Mode B and an on-demand decoded weight cache, validated
against Mode A on tiny and the seven-layer real slice. A CUDA decoder prototype is present but
has not been compiled or run on a GPU; GPU engine integration and measurements remain open.
**M4** has a CPU standard-decoder correctness path and initial QAT primitives. The SmolLM2
Mode A comparison passed against Hugging Face; OLMoE's pretrained comparison and training are
still pending. See `docs/reports/`,
`docs/architecture/format.md` for the on-disk format,
`docs/architecture/deepseek_v41_flash.md` for what the checkpoint contains and
`docs/architecture/engine.md` for the engine design.
* **Mode A.** The engine runs the checkpoint in its shipped formats (FP8 linears, FP4 routed
  experts, BF16/FP32 small tensors). Its kernels (`boilerplate/kernels/torch_ref.py`) are a
  correctness backend: they dequantize each weight and run a floating-point GEMM. They are slow;
  the measured CPU results and hardware are in `docs/reports/M1.md` and `docs/reports/M3.md`.
* Mode A matches the official DeepSeek reference **bit for bit**, with both running on the same
  PyTorch kernels, on a tiny random-init model and on a 7-of-40-layer, 16-of-384-expert slice of
  the real weights without engram. That validates the engine around the kernels, not the full
  model's quality. See `docs/reports/M1.md` for the exact scope.
* **Mode B** losslessly encodes supported tensors on disk and decodes them to the original bytes.
  Decoded tensors and table blocks use an LRU with a configurable byte budget; transient decoding
  and process RSS are not limited by that budget.
* **Mode C** exports and runs stacked ternary weights with TQ1/TQ2 packing, CPU kernels and
  optional GPTQ. A mixed Mode B + C policy is available. Mode D has QAT primitives but no
  complete training or model-export pipeline yet.

## Weight modes

**A** baseline (original FP8/FP4), **B** lossless entropy-coded, **C** stacked residual ternary,
**D** QAT-trained single ternary (planned). Packing: TQ2 (2 bits/weight) and TQ1 (5 trits/byte, 1.6
bits/weight), each plus an FP16 scale per 128 weights (1.725 / 2.125 bits/weight).

What ternary can save here is limited by the source: the routed experts are already FP4 (4.25
bits/weight) and ~40% of the file is FP8 engram tables. The **projected** TQ1 k=1 expert-only
size is 332.9 GB: 508.8 GB source − 296.0 GB source experts + 296.0 × (1.725 / 4.25) GB
ternary experts, using header-derived source sizes and excluding output overhead. This is not
a written file. Stacking two or more ternary terms on the experts saves little or nothing over
FP4. See `docs/architecture/bit_accounting.md`. Ternary weights alone are not new
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

# Export under a precision policy: Mode C (ternary experts), Mode B (lossless), mixed
boilerplate export ./trunc7 ./trunc7_k2 --policy configs/policy/experts_k2.yaml
boilerplate export ./trunc7 ./trunc7_lossless --policy configs/policy/lossless.yaml
boilerplate export ./trunc7 ./trunc7_mixed --policy configs/policy/experts_k2_lossless_rest.yaml
# GPTQ rules (gptq: true) need calibration text
boilerplate export ./trunc7 ./trunc7_gptq --policy my_gptq.yaml --calib-text wiki_train.txt --calib-seqs 512

# Generate
boilerplate generate ./trunc7 "The capital of France is" --max-new-tokens 16
```

Python API:

```python
from boilerplate.engine import Engine
eng = Engine.load("./tiny")
eng.generate([[0, 17, 42]], max_new_tokens=8, temperature=0).tokens
```

For Mode B, `Engine.load(path, mode_b_cache_bytes=256 << 20)` sets the decoded-byte LRU budget;
zero disables retention. The `trace`, `generate` and `bench` commands also accept
`--mode-b-cache-mib` (default 256). Temporary decode buffers, other caches and mmap pages are
outside this limit.

## Layout

```
boilerplate/
  checkpoint/   config/index/header reading, FP8/FP4 block formats
  config/       ModelSpec derivation, tensor roles, size report
  engine/       engine: config, mmap weights (Modes A/B/C), model, paged KV cache, engram, API
  format/       precision policies, export + manifest, Mode B entropy codec, GPTQ calibration
  kernels/      torch_ref (official kernels' numerics) and the C++ CPU extension (ternary GEMM, Huffman)
  qat/          BitLinear, local checkpoint/resume and top-k distillation primitives
  ternary/      TQ1/TQ2 packing, stacked-ternary and GPTQ fitting, ternary linear
  reference/    traces, comparison, official + HF backends
  testing/      tiny checkpoint writer, truncated real-checkpoint slicer
third_party/    official DeepSeek V4.1 inference code (MIT), the trusted reference
configs/        source checkpoint id, precision policies (configs/policy/)
docs/           architecture notes and milestone reports
tests/
```

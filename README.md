# Boilerplate

A ternary-first inference engine for MoE transformer LLMs, plus a pipeline that turns
`deepseek-ai/DeepSeek-V4.1-Flash-1017` into a real ternary (~1.6–2 bits/weight) model.

Weight modes: **A** baseline (original FP8/FP4), **B** lossless entropy-coded, **C** stacked
residual ternary (k = 1–4 terms), **D** QAT-trained single ternary. Packing: TQ2 (2 bits/weight)
and TQ1 (5 trits/byte, 1.6 bits/weight).

Status: **M0** (skeleton, config parser, reference harness) is done. See `docs/reports/M0.md`
and `docs/architecture/deepseek_v41_flash.md` for what the checkpoint actually contains.

## Quick start

```bash
pip install -e ".[hf,dev]"
pytest

# Derive the architecture from a checkpoint (local dir or Hub id; headers via range requests)
boilerplate inspect deepseek-ai/DeepSeek-V4.1-Flash-1017 --json spec.json

# Tiny random-init checkpoint in the exact V4.1 on-disk format
boilerplate make-tiny ./tiny

# Reference trace with the official code (CPU kernel shim by default)
huggingface-cli download deepseek-ai/DeepSeek-V4.1-Flash-1017 --include "inference/*" --local-dir ref
boilerplate reference official ./tiny --code-dir ref/inference --out traces/tiny_ref
boilerplate compare traces/tiny_ref traces/candidate
```

## Layout

```
boilerplate/
  checkpoint/   config/index/header reading, FP8/FP4 block formats
  config/       ModelSpec derivation, tensor roles, size report
  reference/    traces, comparison, official + HF backends, CPU kernel shim
  testing/      tiny checkpoint writer
configs/        source checkpoint id
docs/           architecture notes and milestone reports
tests/
```

# M4: small-model ternary experiment

Status: **in progress**. The [approved plan](../plans/small_model_ternary.md) and its
pre-registered success criterion remain unchanged. No pretrained-model training or GPU
throughput probe has been run.

## Built

| Piece | Where | Status |
|---|---|---|
| QAT primitive | `boilerplate/qat/bitlinear.py` | Per-row/group absmean ternary weights, per-token int8 activations, straight-through gradients, warmup level, and TQ1/TQ2 packed terms. |
| Primitive checks | `tests/test_qat.py` | Tiny synthetic tensors only; checks absmean including a short group, STE gradients, warmup endpoints and packed-linear parity. |
| Local checkpoint | `boilerplate/qat/checkpoint.py`, `tests/test_qat_checkpoint.py` | Atomic single-process model, optimizer, warmup and RNG state save/restore, checked by exact continuation of a tiny synthetic run. |
| Offline distillation primitive | `boilerplate/qat/distill.py`, `tests/test_qat_distill.py` | Store teacher top-k log probabilities and a tail bucket; compute the corresponding KL and a blended CE/KL loss. |

These primitives are not yet a training pipeline or a complete Mode D model exporter. The
caller must save and restore data-loader/sampler state; the local checkpoint permits storing
caller-owned trainer state but does not itself restore worker processes. Offline teacher-data
generation, routing diagnostics, distributed checkpoint/resume and training, and GGUF export
are not yet implemented.

## Measurements

On an AMD EPYC 9V74 cloud CPU (4-core quota, no GPU), CPU-only PyTorch 2.15 and two pinned math
threads, five synthetic QAT primitive tests passed. The packed-linear parity checks use
`atol=1e-3, rtol=1e-3` against the PyTorch engine fallback. The first run had **one failure and
four passes**: the activation absmax scale unintentionally contributed a gradient alongside
the identity STE. Detaching the complete quantized activation fixed the gradient, and the
repeat passed **five tests**. The single-process checkpoint continuation test and two synthetic
coarsened/full-vocabulary distillation tests also passed. These are correctness checks, not
quality or speed results. The QAT branch's full test set passed **141 tests**, with **11 CUDA
tests skipped**; the distillation checks were rerun after adding offline tensor device handling.

Independent work in progress has downloaded the pinned SmolLM2 checkpoint, converted a BF16
GGUF file, and compared a standard-decoder prototype against Hugging Face. For 20 distinct
16-token prompts, all 64 greedy tokens per prompt matched and every one of 32 decoder-block
minimum cosines exceeded 0.999999 on the same AMD CPU with four threads. The implementation,
all runs, file accounting and per-layer numbers will accompany the following standard-decoder
change; they are not part of this QAT primitive change.

## Not done

The CPU baselines, PTQ ternary exports, quality evaluations and speed comparisons have not been
completed. The SmolLM2 shakedown needs an approved GPU budget and throughput probe; OLMoE QAT
needs an approved multi-GPU budget. The pre-registered verdict is **untested** without the
OLMoE QAT and conventional baseline runs.

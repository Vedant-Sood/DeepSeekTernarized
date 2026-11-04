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
| Standard CPU decoder | `boilerplate/engine/standard.py`, `boilerplate/engine/engine.py`, `boilerplate/cli.py` | Config-driven Llama and OLMoE attention, normalization, RoPE, dense/MoE blocks, A/B/C linear loading, API dispatch and CPU benchmarking. |
| Standard-decoder checks | `tests/test_standard_engine.py` | Tiny random Llama/OLMoE comparisons against Hugging Face, including greedy tokens, Mode B byte parity and a mixed B/C export. |
| llama.cpp baseline | External checkout `ggml-org/llama.cpp` at `149f5641b931aaf33697d70f9abc90c3e5300247` | Pinned source; CPU quantize, importance-matrix, perplexity and benchmark tools built. |

These primitives are not yet a training pipeline or a complete Mode D model exporter. The
standard path is not yet validated on the pretrained OLMoE checkpoint. It uses a simple tensor
KV cache and does not yet have a GPU path or paged KV. Offline teacher-data generation, routing
diagnostics, distributed checkpoint/resume, distributed training and GGUF export are not yet
implemented. The caller must save and restore data-loader/sampler state; the local checkpoint
permits storing caller-owned trainer state but does not itself restore worker processes.

## Measurements and source facts

On an AMD EPYC 9V74 cloud CPU (4-core quota, no GPU), CPU-only PyTorch 2.15 and two pinned math
threads, the five synthetic QAT primitive tests passed. The packed-linear parity checks use
`atol=1e-3, rtol=1e-3` against the PyTorch engine fallback. The first run had **one failure and
four passes**: the activation absmax scale unintentionally contributed a gradient alongside
the identity STE. Detaching the complete quantized activation fixed the gradient, and the
repeat passed **five tests**. These are correctness checks, not quality or speed results.
The separate single-process checkpoint continuation test also passed on the same CPU with two
threads; its input and model were synthetic. Two synthetic distillation tests passed, including
the coarsened top-k/tail KL and the full-vocabulary KL. The QAT branch suite passed **141 tests**
with **11 CUDA tests skipped**; the distillation checks were rerun after offline tensor device
handling was added.

Six tiny random-model standard-decoder tests passed on that CPU with two threads: both Llama and
OLMoE matched Hugging Face to the tests' FP32 cosine threshold of 0.9999 and produced the same
greedy tokens for two prompts of eight generated tokens per architecture; their Mode B tensors,
layer traces and logits matched Mode A byte-for-byte, a mixed B/C Llama export ran, and the
`Engine.load` dispatcher and CLI benchmark ran. An earlier run had **one failure and two passes**
because the test fixture assumed a fused expert
file layout; the installed Hugging Face version had already saved the original per-expert
layout. The fixture now accepts either layout. These are tiny random models, not the required
20-prompt, 64-token comparisons on SmolLM2 or OLMoE.

The Hub metadata and safetensors headers were inspected at the pinned revisions below; these
are source facts, not measured local model-file sizes:

| Model | Pinned revision | Checkpoint dtype |
|---|---|---|
| `HuggingFaceTB/SmolLM2-360M` | `f8027fd0eaeea54caa13c31d31b9fdc459c38b49` | BF16 |
| `allenai/OLMoE-1B-7B-0125` | `9b0c1aa87e34a20052389dce1f0cf01da783f654` | F32, all six shard headers |

The plan calls the OLMoE source BF16, whereas the pinned published checkpoint's headers are
F32. A BF16 conversion would be a distinct, lossy source conversion and has not been run. The
plan's success criterion is unchanged; no F32 results are presented as BF16 results.

The pinned SmolLM2 files were subsequently downloaded. `model.safetensors` is **723,674,912
bytes by stat**, SHA-256 `7aaff6661428bed033abba9522bec81938678642cca3181fe752b6ca9e1e540f`,
matching the Hub metadata. The six checkpoint/tokenizer files total **725,784,757 bytes by
stat**, excluding the Hub's local cache metadata. Role payload bytes and parameter counts are
from the written safetensors header; the whole-file figure also includes the header and the
other five files. The embedding is tied to the LM head and counted once.

An initial 20-prompt SmolLM2 reference comparison was interrupted by an environment restart
after 24 of 64 generated tokens; all 20 generated prefixes matched through that point. It did
not write final hidden-state metrics. The second attempt completed using 20 distinct,
16-token windows from the pinned WikiText-2 raw test text, followed by 64 greedy tokens per
window. The tokenizer SHA-256 is
`9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c`, and the saved prompt-ID
file SHA-256 is `87557c41aeda561b5b513b7fe3bc3b8e35a1f17e861e2b438851163c58ae981a`.

| SmolLM2 FP32 Mode A vs Hugging Face | Result |
|---|---:|
| Prompts with all 64 greedy tokens identical | 20 / 20 |
| Matching greedy tokens | 1,280 / 1,280 |
| Minimum cosine over captured HF hidden states | 0.9999991059 |

This run used the AMD EPYC 9V74 CPU with four pinned math threads, PyTorch 2.15 CPU and
transformers 5.22.1. The captured HF hidden states include embeddings, intermediate block
outputs and the final normalized state. A further run explicitly captured each of the 32 block
outputs before final normalization. It again matched **20 / 20** complete greedy continuations;
the minimum cosine at every layer is below. The first complete run took 29.495 s, and the
block-capture run took 623.952 s while other CPU, compilation and download jobs were active.
Both times include loading the two models and are not standalone throughput measurements.

| Layer | Min cosine | Layer | Min cosine | Layer | Min cosine | Layer | Min cosine |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.999999344 | 8 | 0.999999523 | 16 | 0.999999404 | 24 | 0.999999344 |
| 1 | 0.999999344 | 9 | 0.999999523 | 17 | 0.999999523 | 25 | 0.999999285 |
| 2 | 0.999999344 | 10 | 0.999999106 | 18 | 0.999999464 | 26 | 0.999999344 |
| 3 | 0.999999523 | 11 | 0.999999404 | 19 | 0.999999464 | 27 | 0.999999404 |
| 4 | 0.999999404 | 12 | 0.999999523 | 20 | 0.999999523 | 28 | 0.999999285 |
| 5 | 0.999999464 | 13 | 0.999999523 | 21 | 0.999999344 | 29 | 0.999999106 |
| 6 | 0.999999404 | 14 | 0.999999523 | 22 | 0.999999464 | 30 | 0.999999344 |
| 7 | 0.999999523 | 15 | 0.999999404 | 23 | 0.999999523 | 31 | 0.999999523 |

The first SmolLM2 BF16 GGUF conversion attempt did not produce a model: the converter could not
import `sentencepiece`. Version 0.2.2 was installed and the second attempt produced a
**725,553,280-byte** file by stat. The file stores matrices in BF16 and
normalization tensors in FP32; tensor bytes come from GGUF metadata, and the whole-file figure
includes its metadata and tokenizer.

The five conventional quantizations completed; the sizes are from `stat` of the written files.

| SmolLM2 GGUF | File bytes | Whole-file b/w | Source whole-file b/w |
|---|---:|---:|---:|
| BF16 | 725,553,280 | 16.0423 | 16.0474 |
| Q8_0 | 386,404,480 | 8.5435 | 16.0474 |
| Q4_K_M | 270,590,080 | 5.9828 | 16.0474 |
| Q3_K_M | 234,686,080 | 5.1890 | 16.0474 |
| Q2_K | 218,673,280 | 4.8349 | 16.0474 |
| IQ2_XS with importance matrix | 207,998,400 | 4.5989 | 16.0474 |

The GGUF tensor metadata gives the following role-level bits per weight, including each
quantization format's stored scales and the retained high-precision tensors:

| Role | Source b/w | BF16 | Q8_0 | Q4_K_M | Q3_K_M | Q2_K | IQ2_XS |
|---|---:|---:|---:|---:|---:|---:|---:|
| Embedding / tied LM head | 16.0000 | 16.0000 | 8.5000 | 8.5000 | 8.5000 | 8.5000 | 8.5000 |
| Attention | 16.0000 | 16.0000 | 8.5000 | 5.6875 | 5.0039 | 4.5000 | 4.5000 |
| Dense MLP | 16.0000 | 16.0000 | 8.5000 | 5.5104 | 4.5208 | 4.1458 | 3.7839 |
| Normalization | 16.0000 | 32.0000 | 32.0000 | 32.0000 | 32.0000 | 32.0000 | 32.0000 |
| **Whole checkpoint** | **16.0474** | **16.0423** | **8.5435** | **5.9828** | **5.1890** | **4.8349** | **4.5989** |

The first importance-matrix attempt exited with a floating-point exception: passing
`--output-frequency 0` makes the pinned llama.cpp version take a remainder by zero. The
separate retry completed with `--output-frequency 16` and 16 calibration chunks of 2,048
tokens from the pinned WikiText-2 training split; its file is **2,205,664 bytes by stat**.
The training-text SHA-256 is `aee724fa58bfbdeb3fc6803297fb6bab27b203d7c40b39ddef9b9770e5d52fe5`.
This is the llama.cpp importance matrix, not the 512-sequence GPTQ calibration required for
the ternary arms.

For the pinned WikiText-2 raw test text, the source tokenizer and GGUF tokenizer produced the
same **304,978 token IDs** with BOS insertion and special-token parsing disabled. The test-text
SHA-256 is `696cca6b65a171b0a358a4be6732cdfdf2dd6164a32e20fd70e3c13fc4dfae83`. Full
llama.cpp perplexity runs are in progress at context 2,048; its default scorer evaluates the
second half of each complete context window.

| SmolLM2 source / Mode A role | Parameters | Payload bytes | Source b/w | Mode A b/w |
|---|---:|---:|---:|---:|
| Embedding / tied LM head | 47,185,920 | 94,371,840 | 16.0000 | 16.0000 |
| Attention | 78,643,200 | 157,286,400 | 16.0000 | 16.0000 |
| Dense MLP | 235,929,600 | 471,859,200 | 16.0000 | 16.0000 |
| Normalization | 62,400 | 124,800 | 16.0000 | 16.0000 |
| **Whole checkpoint, including non-tensor files** | **361,821,120** | **725,784,757** | **16.0474** | **16.0474** |

## Experiment status

All quality, memory and throughput measurements below are **not run** or in progress. SmolLM2
GGUF file sizes are measured above; OLMoE baselines and PTQ exports need further conversion and
evaluation. QAT and GPU comparisons also need the hardware and training approvals in the task.
No model-file size or speed is inferred from the synthetic tests.

| Arm, for each applicable model | Status |
|---|---|
| Original source and BF16 comparison | SmolLM2 source size measured above; complete evaluations and OLMoE source conversion not yet performed |
| llama.cpp Q8_0, Q4_K_M, Q3_K_M, Q2_K, IQ2_XS | SmolLM2 files measured above; OLMoE files and quality/speed evaluations not run |
| Absmean TQ1 / TQ2, plain / GPTQ | not run: full-model PTQ exports and evaluation not yet performed |
| Mode C TQ1 k=2, plain / GPTQ | not run: full-model PTQ exports and evaluation not yet performed |
| SmolLM2 QAT shakedown | not run: GPU budget and throughput probe require owner approval |
| OLMoE QAT at 1 B / 3 B / 10 B tokens, TQ1 / TQ2 | not run: multi-GPU hardware and training budget require owner approval |

For every arm except the SmolLM2 source and GGUF sizes reported above, file bytes and effective
bits per weight are **not run**. For every arm, CPU RSS and GPU VRAM; batch-1 and
batch-8 prefill/decode throughput on CPU and GPU; WikiText-2 and held-out pretraining-mix
perplexity; MMLU, ARC-Challenge, HellaSwag, PIQA, WinoGrande and GSM8K; and OLMoE router
overlap and expert-load entropy are **not run** for the reasons above.

The pre-registered verdict is **untested**: there is no OLMoE QAT run or evaluated conventional
baseline to compare. In particular, the primitive tests do not establish model quality, file
size or speed.

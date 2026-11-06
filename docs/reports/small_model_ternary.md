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

*Superseded on 2026-11-05: the training pipeline, Mode D export, teacher data, routing
diagnostics, distributed checkpointing and GGUF export now exist; see "Follow-up, 2026-11-05"
at the end. The original paragraph follows.* These primitives are not yet a training pipeline or a complete Mode D model exporter. The
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

## Follow-up, 2026-11-04: SmolLM2 post-training ternary arms

Built in this change (see the table in PR #22): `fit: absmean` policies (the BitNet b1.58 quantizer
that QAT trains against), GPTQ calibration through the standard decoder, routing-drift metrics,
FSDP2 sharding with a 2-process `gloo` QAT dry run (exact resume), and `boilerplate gguf` for
llama.cpp `TQ1_0`/`TQ2_0`.

**Setup.** Source `HuggingFaceTB/SmolLM2-360M` at the pinned revision above. Every projection in
the 32 blocks (q/k/v/o, gate/up/down: 314,572,800 weights) is ternary. The tied embedding/LM head
(BF16) and norms stay in the source format, as the plan says. Group 128, FP16 scales, int8
per-token activations. GPTQ Hessians: 128 non-overlapping 512-token windows of the pinned
WikiText-2 train text (SHA-256 `aee724fa…`). They were collected on the unquantized model in
one pass (not the sequential quantized-input variant). Calibration took 300.9 s and no weight
fell back to the plain fit.

**Perplexity protocol.** Pinned WikiText-2 raw test rows joined with blank lines (SHA-256
`696cca6b…`, 304,978 tokens, no BOS), non-overlapping 2,048-token windows, second half of each
window scored, as llama.cpp `llama-perplexity` does. **Only the first 40 of the 148 windows
(40,920 scored tokens) were evaluated.** A container restart killed the first full-length runs
at 40/148 windows of the source. The reruns were cut to 40 windows to fit in the session, so
these are subset perplexities, not the plan's full-test number. Hardware: Intel Xeon @ 2.10 GHz
cloud CPU, 4 cores, 4 pinned threads, PyTorch 2.15 CPU, FP32 compute. Ternary layers use the
`torch` backend: int8-quantized activations times the cached dequantized trits. It computes the
same values as the C++ kernel, up to FP32 summation order.

| Arm | Fit | Ternary b/w (scales incl.) | Whole model b/w | Export dir bytes (stat) | WikiText-2 ppl, 40 windows |
|---|---|---:|---:|---:|---:|
| Source (BF16, FP32 compute) | — | — | 16.0474 | 725,784,757 † | **9.577** |
| Absmean TQ1 k=1 | absmean | 1.7313 | 3.5945 | 165,003,247 | 584,405 |
| Absmean TQ2 k=1 | absmean | 2.1313 | 3.9423 | 180,731,888 | same trits as TQ1 ‡ |
| TQ1 k=1 | least squares | 1.7313 | 3.5945 | 165,002,342 | 937,533 |
| TQ1 k=1 + GPTQ | GPTQ | 1.7313 | 3.5945 | 165,002,122 | 18,364 |
| Mode C TQ1 k=2 | least squares | 3.4625 | 5.0997 | 233,280,070 | 108,276 |
| Mode C TQ1 k=2 + GPTQ | GPTQ | 3.4625 | 5.0997 | 233,279,850 | **21.083** |

† The six checkpoint/tokenizer files measured earlier in this report. Export directories also
contain `manifest.json` and the index. "Whole model b/w" uses the manifest's payload bytes over
all 361,821,120 parameters, with the BF16 embedding included.
‡ TQ2 stores exactly the same trits and scales as TQ1, so its outputs are identical; only the
packing (size) differs. Not re-run.

What this says:

* **Plain ternary PTQ of every linear destroys a 360M model.** Absmean (the QAT quantizer
  without training), the least-squares single term and plain two-term Mode C all give
  perplexities of 10^5 to 10^6. For reference, a uniform distribution over the 49,152-token
  vocabulary has perplexity 49,152. On the first two windows, the C++ kernel backend gave the
  same order of magnitude for absmean (2.9 M vs 3.2 M with the torch backend). Both are
  collapsed, and the difference is not meaningful there. A diagnostic run on 512 tokens showed
  the hidden-state cosine against the source falling from 0.95 (mean, layer 0) to 0.34 (layer
  31) for plain k = 2. Per-matrix relative weight error was 0.22 (down_proj) and 0.30 (q_proj).
* **GPTQ is what makes k = 2 usable**: 21.1 against 9.58 for the source, 2.2× the source
  perplexity. That is the same direction as M2's finding on DeepSeek experts (plain k = 2 not
  close enough, GPTQ needed), here at full-model scale. GPTQ does not rescue a single term
  (18,364).
* Size context: the k = 2 + GPTQ arm (233.3 MB on disk) is **larger** than llama.cpp `Q2_K`
  (218.7 MB) and `IQ2_XS` (208.0 MB) from the table above. One reason is that the plan keeps
  the tied embedding in BF16 (94.5 MB, 41% of the file), while llama.cpp stores it at 8.5 b/w.
  Quality comparison with those files needs llama.cpp perplexity on the same 40 windows. That
  is **not run**: building and running llama.cpp was not permitted in this environment, so it
  has to run on another machine.
* The verdict criterion is about QAT, not PTQ, and remains **untested**. These arms show what
  training has to recover: a single ternary term without training is unusable at this scale.

GGUF: with the `std_absmean_tq1_g256` policy (group 256 = llama.cpp block),
`boilerplate gguf` converted the 32 `down_proj` tensors (input dim 2,560) to `TQ1_0` exactly.
The other 192 projections have input dim 960, which is not a multiple of 256, so llama.cpp's
TQ types cannot hold them. A llama.cpp ternary comparison on identical weights is therefore only
meaningful on OLMoE (2,048 / 1,024).

Not run in this follow-up, with reasons: the remaining 108 windows (session time); CPU RSS and
speed for the PTQ arms (the timings above share the CPU with other jobs and include FP32
dequantization); held-out pretraining-mix perplexity and lm-eval tasks (not yet wired up); all
OLMoE arms (the F32 → BF16 source question is waiting on Vedant; the F32 checkpoint is about
28 GB, which also exceeds this container's 15 GB RAM).

## Experiment status

All quality, memory and throughput measurements below are **not run** or in progress. SmolLM2
GGUF file sizes are measured above; OLMoE baselines and PTQ exports need further conversion and
evaluation. QAT and GPU comparisons also need the hardware and training approvals in the task.
No model-file size or speed is inferred from the synthetic tests.

| Arm, for each applicable model | Status |
|---|---|
| Original source and BF16 comparison | SmolLM2 source size measured above; complete evaluations and OLMoE source conversion not yet performed |
| llama.cpp Q8_0, Q4_K_M, Q3_K_M, Q2_K, IQ2_XS | SmolLM2 files measured above; OLMoE files and quality/speed evaluations not run |
| Absmean TQ1 / TQ2, plain / GPTQ | SmolLM2: exported, sizes and 40-window WikiText-2 ppl in the follow-up above (absmean has no GPTQ variant; GPTQ is reported for the least-squares single term); OLMoE not run |
| Mode C TQ1 k=2, plain / GPTQ | SmolLM2: exported, sizes and 40-window WikiText-2 ppl above; OLMoE not run |
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

## Follow-up, 2026-11-05: code complete for the gated runs

Everything the plan needs before a GPU is approved is now in the repository and tested on tiny
random models on CPU. **Nothing in this section is a measurement of a pretrained model**: no QAT
run, throughput probe or GPU kernel run has happened.

| Piece | Where | Tested by |
|---|---|---|
| Training loop from one YAML config: FSDP2 when `WORLD_SIZE > 1`, AdamW (or 8-bit AdamW), LR warmup + cosine, lambda warmup, CE or CE + KL to stored teacher top-k, gradient clipping, JSON-lines logs, routing drift on a fixed probe batch, checkpoints, exact resume, Mode D export at the end and at `export_at_tokens` | `boilerplate/qat/train.py`, `boilerplate qat-train` | `tests/test_qat_train.py` (export parity with the engine, exact resume, OLMoE with offline teacher, snapshots) |
| Approval gate: runs over 20 optimizer steps refuse to start without `approval:` in the config | `train.py` | `test_approval_gate` |
| Token files, sampler with resumable state and rank split; offline teacher top-k shards | `qat/data.py`, `qat/teacher.py`, `qat-tokenize`, `qat-teacher` | `tests/test_qat_train.py` |
| OLMoE F32 → BF16 cast with SHA-256 per shard (decision: cast first, both paths supported) | `checkpoint/cast.py`, `boilerplate cast` | `test_cast_checkpoint_f32_to_bf16` |
| Mode D export (one absmean ternary term per BitLinear) and GGUF `TQ1_0`/`TQ2_0` | `qat/export.py`, `format/gguf_export.py` | `tests/test_qat_train.py`, `tests/test_gguf_export.py` |
| CUDA ternary GEMM/GEMV and grouped expert GEMM (TQ1/TQ2, int8 activations, one launch for all experts) | `kernels/csrc/ternary_cuda_kernel.cu`, `kernels/ternary_cuda.py` | `tests/test_ternary_cuda.py`: reference vs dense and vs the C++ CPU kernel run here; **the CUDA parity tests skip: the kernel has not been compiled or run** (no GPU or toolkit in this container) |
| Compute / memory / cost estimator | `qat/estimate.py`, `boilerplate qat-estimate` | `test_qat_estimate` |
| lm-eval-harness adapter (loglikelihood, rolling, generate_until) | `evals/lm_eval_adapter.py`, `boilerplate lm-eval` | `tests/test_lm_eval_adapter.py` (scores match transformers) |
| Launch scripts | `scripts/launch/smollm2_shakedown.sh`, `scripts/launch/olmoe_qat.sh`, `configs/qat/*.yaml` | `bash -n`; gates checked by hand |

Still not built: expert parallelism (FSDP2 shards experts like any other weight, which is enough
for OLMoE at 4 × 80 GB but not for V4.1), a fused grouped-expert path in the training graph
(`qat/model.py::Experts` loops over experts in Python, which keeps MFU low on MoE), a NEON build of
the CPU kernel (not testable here: no ARM runner or qemu), and BF16 activations in the CUDA kernel
(int8 only; BF16 falls back to the dequantized matmul).

### Compute estimates (planning numbers, not measurements)

From `boilerplate qat-estimate` with the committed configs: training FLOPs = 6 × active
parameters × tokens; the offline teacher pass adds 2 × active × tokens once. SmolLM2-360M has
361.8 M parameters (counted from its safetensors headers). OLMoE-1B-7B-0125: 6.92 B total,
1.28 B active (computed from its `config.json`: 16 layers, hidden 2048, 64 experts of
intermediate 1024, top-8, untied 50,304-token embeddings; the model card says 6.9 B / 1.3 B).
MFU is an assumption until a 20-step probe gives a measured tok/s; MoE training with the
per-expert loop is expected at the low end.

| Run | Tokens | Train FLOPs | Teacher FLOPs | Memory / GPU (est.) | Hardware | Wall clock at MFU 10% / 20% / 40% | GPU-hours at the same |
|---|---|---|---|---|---|---|---|
| SmolLM2 shakedown | 0.5 B | 1.1e18 | 3.6e17 | 2.5 GiB (8-bit AdamW, checkpointing) | 1 × RTX 4060 laptop (~60 TFLOP/s BF16 assumed) | 50 h / 25 h / 13 h | same |
| OLMoE QAT | 1 B | 7.7e18 | 2.6e18 | 19.8 GiB (FSDP over 4) | 4 × H100 80 GB | 5.4 h / 2.7 h / 1.4 h | 22 / 11 / 5 |
| OLMoE QAT | 3 B | 2.3e19 | 7.7e18 | 19.8 GiB | 4 × H100 | 16 h / 8.1 h / 4.1 h | 65 / 32 / 16 |
| OLMoE QAT (snapshots at 1 B, 3 B) | 10 B | 7.7e19 | 2.6e19 | 19.8 GiB | 4 × H100 | 54 h / 27 h / 13.5 h | 216 / 108 / 54 |

The 10 B estimate at MFU 10–20 % (108–216 H100-hours) agrees with the plan's 100–200. Multiply
GPU-hours by the provider's hourly price for cost (`--price`). The OLMoE memory figure leaves room
for a larger micro batch; the plan's 2–4 GPUs are both feasible.

Next, all gated on Vedant: approve the SmolLM2 probe and shakedown budget (then
`scripts/launch/smollm2_shakedown.sh prepare|probe|train|eval`), and the OLMoE budget
(`scripts/launch/olmoe_qat.sh`). The OLMoE baselines and PTQ arms need a machine with more than
this container's 15 GB RAM; llama.cpp runs need a machine where building it is allowed.

## Follow-up, 2026-11-06: pre-GPU fixes and GPU-day tooling

Done before GPU access, so GPU time goes to measurements. Still **no pretrained-model training
and no GPU run**.

* **Real-model CPU smoke.** Three QAT steps of the real SmolLM2-360M ran on CPU (seq 128, offline
  teacher, Mode D export). They found and fixed three problems:
  * The sampler drew windows the teacher had not covered. It now trains only on covered windows,
    and refuses a teacher built for another sequence length or token file.
  * Mode D export wrote embeddings and norms in the FP32 master dtype, which made the file 260 MB
    instead of ~165 MB. They are now written in the source's dtype.
  * The tokenizer read the whole corpus into Python lists. That needs about 280 GB of RAM for
    10 B tokens. It now streams.
* **Export parity on the real model.** Logits of the exported model differ from the training
  graph's by 0.95 at most (4.6 % of the largest logit), and 92 % of greedy tokens agree. Layers
  0–2 agree to 1e-6, and the difference grows from layer 3 onward. This comes from the int8
  activation rounding, not from the export. Scaling the training graph's own input embeddings by
  (1 + 1e-6) changes its logits by 0.70 at most (3.4 %). So agreement to that level is all that can
  be expected after three steps. The tiny-model parity test (2e-3) still holds.
* **Online teacher.** `teacher_model:` runs the BF16 teacher next to the student and takes its
  top-k every step. Offline top-k at k = 32 costs 132 bytes per token, which is about 66 GB for the
  0.5 B-token shakedown and about 1.3 TB for OLMoE at 10 B. All committed configs now use the online
  teacher; `teacher_dir` still works. Online and offline give the same losses (fp16 rounding aside).
* **Data.** `boilerplate qat-data` streams a corpus from Hub datasets by range requests: FineWeb-Edu
  for SmolLM2, and OLMoE-mix-0924 weighted by its subset token counts. It holds out each source's
  last file as a disjoint held-out split. `qat-tokenize` takes that JSONL and puts EOS between
  documents. `scripts/eval/get_wikitext2.py` writes WikiText-2 byte-identically to the text every
  perplexity in this report used.
* **GPU path for the Llama/OLMoE engine.** `device="cuda"` (`--device cuda` on `bench`, `serve`,
  `lm-eval` and the perplexity script) keeps weights resident and runs ternary layers with the CUDA
  kernel. On CPU, the materialized path matches the mmap path. A CUDA parity test is ready for the
  GPU.
* **CUDA kernels compile.** `scripts/dev/cuda_compile_check.sh` builds both CUDA sources for sm_89
  (the RTX 4060) with clang's CUDA front end and NVIDIA's ptxas, without a GPU. The ternary kernel
  uses 68–70 registers with no spills (a spill was fixed). They have still never run.
* **QAT MoE dispatch.** One sort per layer replaces a full scan per expert. Outputs and gradients
  are identical.
* **Routing drift** for any MoE checkpoint against its source: `scripts/eval/routing_drift.py`,
  with routing capture in both engines.
* **`scripts/launch/gpu_day.sh`** runs, in order: environment check, CUDA build, every test
  (the CUDA ones must not skip), kernel benchmark, engine decode benchmark (BF16 vs absmean TQ1,
  batch 1 and 8), and the 20-step throughput probe with the measured estimate. Results go to one
  folder.

### Updated estimates (online teacher)

The online teacher's forward pass now counts inside each step, which adds about one third to the
compute. The shakedown micro batch drops to 2 (with gradient accumulation 16) so the student's and
the teacher's 49k-vocabulary logits fit in 8 GB.

| Run | Memory / GPU (weights + state + teacher) | Wall clock at MFU 10% / 20% / 40% | GPU-hours at the same |
|---|---|---|---|
| SmolLM2 shakedown, 0.5 B, RTX 4060 laptop | 2.9 GiB + activations and logits | 67 h / 34 h / 17 h | same |
| OLMoE 1 B, 4 × H100 | 32.7 GiB | 7.2 h / 3.6 h / 1.8 h | 29 / 14 / 7 |
| OLMoE 3 B, 4 × H100 | 32.7 GiB | 22 h / 11 h / 5.4 h | 86 / 43 / 22 |
| OLMoE 10 B, 4 × H100 | 32.7 GiB | 72 h / 36 h / 18 h | 288 / 144 / 72 |

The 20-step probe in `gpu_day.sh` replaces the MFU assumption with a measured rate.

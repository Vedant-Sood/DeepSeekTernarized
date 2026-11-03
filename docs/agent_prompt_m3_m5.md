# Coding-agent prompt: finish M3, run M4, deliver M5

Paste everything below the line into a coding agent that has this repository checked out.
Written 2026-11-03 against `main` at `600a9a1` (PR #5 merged). If `main` has moved, the agent
must re-check the "Current state" section against the repo before relying on it.

---

## 0. Your role and how to work

You are continuing **Boilerplate**, an inference engine for MoE LLMs with compressed weights, and
its pipeline for turning `deepseek-ai/DeepSeek-V4.1-Flash-1017` into a ternary model. Milestones
M0–M2 are done and M3 is mostly done. Your job is to finish M3, carry out M4 (the small-model
ternary experiment), and carry out M5 (full-model conversion, evaluation, final report) up to the
approval gates in section 2.

Before writing code, read, in this order:

1. `README.md`
2. `docs/architecture/deepseek_v41_flash.md`: what the checkpoint actually contains.
3. `docs/architecture/bit_accounting.md`: what ternary can and cannot save here. **Binding.**
4. `docs/architecture/engine.md` and `docs/architecture/format.md`
5. `docs/reports/M0.md` … `M3.md`: what was built, what was measured, what was left open.
6. `docs/plans/small_model_ternary.md`: the approved plan that **is** M4. **Binding.**
7. `configs/policy/*.yaml`, `boilerplate/format/policy.py`, `boilerplate/ternary/`,
   `boilerplate/kernels/csrc/boilerplate_cpu.cpp`, `boilerplate/engine/`.

Work in small, reviewable pull requests (one per milestone at least, more for large pieces), on
a feature branch, never directly on `main`. Every PR ends with `pytest` green. Every milestone ends
with a short written report in `docs/reports/` that follows the style of the existing ones: what
was built (table with file paths), what was measured (tables, with the hardware stated), what is
not done and why.

## 1. The project in one paragraph

Four weight modes, chosen per tensor by a YAML precision policy: **A** baseline (the checkpoint's
own FP8/FP4/BF16), **B** lossless entropy-coded (bit-identical outputs to A), **C** stacked
residual ternary `W ≈ Σᵢ sᵢ·Tᵢ`, k = 1–4, group scales, optional GPTQ, **D** real ternary (k = 1)
produced by QAT, BitNet b1.58 style. Packing: TQ2 (2 b/w) and TQ1 (5 trits/byte, 1.6 b/w), plus an
FP16 scale per group of 128 (1.725 / 2.125 b/w effective). Routed experts get ternarized;
embeddings, LM head, norms and router stay high precision. Output format: a directory with
`manifest.json` (architecture, per-tensor mode/packing/group/scale dtype/checksum, effective bits
per weight) and sharded tensor files.

## 2. Hard rules (non-negotiable)

**Approval gates.** Stop, write down what you want to run and what it costs, and wait for the
owner (Vedant) to say yes in writing before:

- **Any full-size QAT / Mode D training run of DeepSeek V4.1.** Never launch it without Vedant's
  explicit confirmation. Write the code, the compute estimate and the launch scripts, then stop.
- **Any small-model training run** (SmolLM2, OLMoE), including on Vedant's own laptop GPU, until
  Vedant has approved a GPU budget for it. Unit tests and smoke tests of a few steps on tiny
  random-init models are fine without approval.
- Renting or paying for any hardware or service.
- Downloading the full ~510 GB checkpoint or running the full conversion (it needs a machine
  this container is not; say which machine and why).
- Changing a success criterion or acceptance threshold after seeing results.
- Using `allow_larger` in a policy for anything other than a clearly labelled side experiment.

**Honesty rules.**

- **Never report a number you have not measured.** Projections are allowed only when labelled
  "projected", with the formula and inputs next to them. Sizes come from `stat` of written files.
- **No "lossless ternary".** Re-encoding weights in base 3 does not shrink them. Size comes only
  from entropy coding (Mode B, lossless) or quantization (Modes C/D, lossy).
- No cherry-picking. Report every arm, every run, including failed and bad ones. If a run was not
  done, the cell says "not run" and the reason.
- Effective bits per weight always include scales and the high-precision tensors, per role and
  for the whole file, and always next to the source bits for that role.

**Engineering rules.**

- Don't hardcode architectures. Derive everything from `config.json` and the safetensors headers.
  This also applies to the new standard-decoder path in M4.
- Trusted references: for V4.1, the official DeepSeek `inference/` code vendored at
  `third_party/deepseek_v41_inference/` (HF transformers has no `deepseek_v41`). For SmolLM2 and
  OLMoE, HF transformers.
- Small before big: develop on `boilerplate make-tiny` and the `make-truncated` 7-layer real slice
  before touching the full checkpoint.
- Determinism: CPU comparisons need pinned math threads (`MKL_DYNAMIC=FALSE` is set in
  `boilerplate/__init__.py`; also pin `torch.set_num_threads` / `OMP_NUM_THREADS` explicitly in
  every measurement script and record the value).
- **Size rule for Mode C** (`bit_accounting.md`, enforced in `policy.py`): refuse any setting
  whose effective bits are not below 0.85× the source bits for that role. For the FP4 routed
  experts (4.25 b/w) that allows only **TQ1 k ≤ 2 or TQ2 k = 1**. Defaults: TQ1 k = 1 (size) or
  TQ1 k = 2 (quality). For BF16 sources (SmolLM2, OLMoE) the same rule allows more.
- **Engram tables:** ~203 GB of the 510 GB file are two FP8 engram lookup tables. Every full-model
  result is reported for **both** variants: engram tables kept FP8, and engram tables ternarized.

## 3. Current state (verify against the repo)

**Done and merged.**

- M0: repo skeleton, config parser (`boilerplate inspect`), reference harness, traces and
  layer-by-layer compare.
- M1: Mode A engine (`boilerplate/engine/`), bit-exact with the official reference on the tiny model
  and on a 7-of-40-layer, 16-of-384-expert real-weight slice. Paged KV cache, engram, MoE grouped
  by expert, mmap'd weights. Kernels are the pure-PyTorch correctness backend
  (`boilerplate/kernels/torch_ref.py`): slow (0.064 tok/s decode on the slice, 4 CPU cores).
- M2: TQ1/TQ2 packing, greedy residual fit and GPTQ fit (`boilerplate/ternary/`), C++ CPU ternary
  GEMM (AVX2 add/subtract, scalar fallback), policy + export + manifest, Mode C in the engine.
  Measured on the slice: plain TQ1 k = 2 hidden-state cosine falls to **0.966** after 7 layers, so
  plain k = 2 is *not* close to baseline over 40 layers; **GPTQ is needed** (−38% output error on
  real expert inputs for k = 2). End-to-end GPTQ over the slice has not been run.
- M3 (on CPU): Mode B canonical Huffman codec (12-bit codes, 4096-symbol blocks), C++ encode/decode,
  format, engine support. **Bit-identical to Mode A** on the slice and the tiny model. Measured
  ratios: FP4 experts 0.977×, FP8 0.83×, BF16 0.67×; whole release projected ≈ 449.5 GB (0.88×).
  Decode 532 MB/s on CPU.

**Not done (gaps you are inheriting).**

- No CUDA code at all: no GPU ternary GEMV/GEMM, no grouped expert GEMM, no GPU Huffman decode.
- No NEON path; AVX-512 only via auto-vectorization.
- Mode B decodes a whole tensor on first use and keeps it: saves disk, *costs* memory (peak RSS
  12.1 GB vs 8.4 GB for Mode A on the slice).
- Mode B + Mode C in one policy is untested.
- No OpenAI-compatible HTTP server (the CLI and Python API exist).
- No explicit expert cache / offload beyond OS page cache over mmap.
- No QAT code at all.
- No engine path for standard decoders (Llama-style dense, OLMoE-style MoE).
- `README.md` still says "No ternary code exists yet" and "Planned weight modes"; it is stale.

**Projected full-model sizes (from shard headers × measured b/w; projections, not files):**
release 508.8 GB; experts TQ1 k = 2: 453.1 GB; experts TQ1 k = 1: 332.9 GB; experts + engram TQ1
k = 2: 335.1 GB; experts + engram TQ1 k = 1: 172.5 GB. Ternary experts make the file ~1.5×
smaller, not 5–8×, because the experts are already FP4.

## 4. Hardware: what runs where

| Machine | What it is | Use it for |
|---|---|---|
| Cloud dev container | 4 CPU (AVX2/AVX-512), 15 GB RAM, **no GPU**, ~29 GB free disk, HF Hub reachable | All CPU code, tests, the tiny model, the 7-layer slice (~6 GB; the reference on it needs ~12 GB RAM), llama.cpp baselines and CPU evals of SmolLM2/OLMoE |
| Vedant's laptop | RTX 4060 **8 GB VRAM**, 32 GB RAM, likely Windows → WSL2 | CUDA kernel development and tests, consumer-GPU speed numbers, the SmolLM2-360M QAT shakedown (~0.3–0.5 B tokens, 8-bit Adam, gradient checkpointing) **once approved**. Not OLMoE training |
| HF ZeroGPU (Space `VedJSood/boilerplate-kernel-check`, private) | Short, time-limited GPU calls on a MIG slice | Quick kernel-vs-reference numerics checks. Not development, not training. `sparse_attn` with h = 64 exceeds its shared memory |
| Rented multi-GPU | Not available yet | OLMoE QAT, full-model GPTQ/eval, full-model QAT. **Each needs Vedant's approval and budget** |

Anything that needs a machine you don't have: build it, test it at small scale, write the exact
command, the hardware it needs and a cost/time estimate, and stop at the gate.

## 5. M3: finish Mode B

M3's correctness goal (bit-identical to Mode A) is met on CPU. What remains is making Mode B
useful at runtime and on GPU.

### Tasks

1. **Bounded decode on CPU.** Replace "decode whole tensor on first use and keep it" with
   decode-on-demand per layer / per expert into an LRU cache with a configurable byte budget. Peak
   RSS must no longer exceed Mode A's by more than that budget.
2. **GPU Huffman decode kernel** (CUDA): one thread block per 4096-symbol block, the 4096-entry
   decode table in shared memory, per-block bit offsets as already stored in the format. Decode
   per layer (or per routed expert) just before the FP8/FP4 GEMM, into a reused scratch buffer,
   DFloat11 style. Fusing decode into the GEMM is optional; measure first whether it matters.
   Build it as a torch extension next to the CPU one, with the same Python interface and a clean
   fallback when CUDA is absent.
3. **Mixed Mode B + Mode C policy.** Add `configs/policy/experts_k2_lossless_rest.yaml` (Mode B
   for everything not ternarized, Mode C for experts) and a test that exports, loads and runs it
   on the tiny model, with the Mode B tensors byte-identical to the source.
4. **Optional:** rANS for the e8m0 scales and FP8 (the last ~1–2%). Only if it is cheap; it is not
   required for acceptance.
5. Fix the stale lines in `README.md` (current status, modes that exist now).

### M3 acceptance criteria

- `pytest` green, including new tests: GPU decode output byte-identical to the CPU decoder on all
  dtypes the codec handles (uint8 / FP8 / packed FP4 / BF16 / FP16 high byte), skipped cleanly
  without CUDA.
- Engine Mode B logits **bit-identical** to Mode A on the tiny model and the 7-layer slice, with
  both the CPU bounded-cache path and the GPU decode path (GPU run on the RTX 4060 or ZeroGPU).
- Measured and reported: GPU decode throughput (GB/s of original bytes) on named hardware; peak
  RSS / VRAM of Mode B vs Mode A with the cache budget stated; prefill/decode tok/s for both.
- `docs/reports/M3.md` updated with a dated section for these results; the "Not done" list updated.

## 6. M4: the small-model ternary experiment

M4 **is** `docs/plans/small_model_ternary.md` (it replaced the old "QAT on a small model"). Follow
that plan's arms, measurements and **pre-registered success criterion** exactly. Do not change the
criterion after seeing results.

### Tasks

1. **Standard-decoder engine path.** Load and run `HuggingFaceTB/SmolLM2-360M` (dense) and
   `allenai/OLMoE-1B-7B-0125` (64 experts, top-8) from their `config.json`: RMSNorm, GQA,
   QK-norm where configured, RoPE from the config, SwiGLU, softmax top-k routing with the config's
   normalization. Reuse the existing `QLinear`/policy/export machinery so Modes A, B, C, D all
   work on these models. Validate against HF transformers.
2. **CUDA ternary kernels.** Multiply-free ternary GEMV (decode) and GEMM (prefill), unpack and
   group scale fused in, for TQ1 and TQ2, int8 per-token absmax activations (BF16 option), plus a
   grouped GEMM for MoE experts. Same Python interface as the CPU kernel. Test against the
   dequantized matmul within tolerance. Develop and measure on the RTX 4060.
3. **NEON path** for the CPU ternary kernel (carried over from M2). Test it under
   `qemu-aarch64` or on an ARM runner if available; if neither is available, say so and leave the
   scalar fallback.
4. **Baselines (CPU, no approval needed).** Pin a llama.cpp commit. Produce `Q8_0`, `Q4_K_M`,
   `Q3_K_M`, `Q2_K`, `IQ2_XS` (with an importance matrix) for both models. Measure file bytes,
   peak memory, prefill/decode tok/s, WikiText-2 perplexity and a held-out slice of the pretraining
   mix (fixed 2048 context, same tokenizer), and the lm-eval-harness tasks in the plan.
5. **PTQ ternary arms (no training).** Absmean ternary TQ1 and TQ2 (k = 1), Mode C TQ1 k = 2, each
   plain and with GPTQ (512 calibration sequences). Same measurements.
6. **QAT module** (`boilerplate/qat/`):
   - `BitLinear`: forward uses absmean ternary weights and 8-bit per-token absmax activations;
     backward is the straight-through estimator; BF16 master weights.
   - Lambda warmup that blends full precision → fully quantized over the first N steps.
   - Optional distillation: KL to the BF16 teacher's logits plus CE. On 8 GB VRAM, precompute and
     store the teacher's top-k logits offline instead of keeping the teacher resident.
   - Router frozen by default; log per-layer expert load entropy and top-k overlap with the BF16
     model during training (routing drift).
   - Checkpoint/resume (bit-identical continuation on resume), configurable token budget, a
     config file per run, FSDP + expert parallelism designed in. Test the distributed path with a
     2-process CPU (`gloo`) smoke test on a tiny model.
   - Export to Boilerplate TQ1/TQ2 (Mode D) **and** to llama.cpp GGUF `TQ1_0`/`TQ2_0`, so kernels
     can be compared on identical weights.
7. **SmolLM2-360M shakedown on the RTX 4060 (GATED).** Before launching: post the dataset,
   token budget, sequence length, VRAM estimate and a wall-clock estimate derived from a measured
   short throughput probe (ask for the probe in the same request if it has not been approved).
   Wait for Vedant's go-ahead. Then run, checkpoint regularly, and report the loss curve and every
   measurement from the plan.
8. **OLMoE QAT at 1 B / 3 B / 10 B tokens (GATED).** Write launch scripts, a dry run on a tiny
   OLMoE-shaped random config, and the compute/cost estimate (the plan's estimate is ~1e20 FLOPs,
   roughly 100–200 H100-hours, 2–4 × 80 GB with FSDP; refine it from measured throughput). Stop
   and ask for the budget. Do not train without it.
9. **Speed comparison.** Same machine per comparison, batch 1 and 8: Boilerplate vs llama.cpp on
   the 4-core CPU and on the RTX 4060, for BF16, `Q4_K_M`, `Q2_K`/`IQ2_XS` and ternary
   (ours vs llama.cpp `TQ*_0` on identical weights).
10. **Report** `docs/reports/M4.md` (or `small_model_ternary.md` as the plan names it): every arm,
    every measurement, and the verdict against the pre-registered criterion.

### M4 acceptance criteria

- Standard-decoder Mode A matches HF transformers on SmolLM2 and OLMoE: per-layer hidden-state
  cosine ≥ 0.9999 in FP32 and identical greedy tokens on at least 20 prompts × 64 tokens.
- CUDA ternary kernels match the dequantized reference matmul within tolerance (state it) for both
  packings, int8 and BF16 activations, groups 64/128/256 and ragged K; grouped expert GEMM matches
  a per-expert loop.
- QAT tests: quantizer matches a reference absmean implementation; STE gradients correct;
  lambda = 0 reproduces the full-precision model; lambda = 1 training-graph forward matches the
  engine running the exported Mode D weights within tolerance (export parity); resume is exact.
- Every arm × metric cell in the plan's tables is filled with a measured number or "not run: reason".
- The shakedown has run (after approval) and is reported with its loss curve, or the report states
  it is waiting on approval.
- The verdict against the success criterion is stated plainly, positive or negative. If OLMoE QAT
  was not approved, the report says the criterion is untested and what the shakedown suggests,
  without dressing that up as the verdict.
- **If the criterion fails at the largest run budget: stop and report.** Do not start any
  full-model QAT planning beyond what M5 already asks for until Vedant decides.

## 7. M5: full-model conversion, evaluation, final report

### Tasks

1. **Streaming converter at full scale.** Shard by shard, never the whole model in RAM; dequantize
   FP8/FP4 blocks to BF16 on the fly only where a policy needs it; resumable after interruption;
   SHA-256 per tensor in the manifest. Test it on the tiny model and the slice, then write down the
   disk/RAM/time the full run needs (it does not fit this container: ~510 GB source plus outputs).
   **Stop for approval before the full download and conversion.**
2. **Full-scale GPTQ for Mode C experts.** Calibration through the Mode A engine with 512
   sequences, Hessians collected per layer (`--calib-layers`) so memory stays bounded. Many of the
   384 experts per layer will get few calibration tokens: set a minimum token count, fall back to
   the plain fit (or a dampened Hessian) below it, and report how many experts took each path. The
   plain fit alone was ~24 h on the 4-core CPU for all experts; plan this on GPU.
3. **Variants to produce** (each × both engram choices where it applies):

   | Variant | Experts | Engram tables | Rest |
   |---|---|---|---|
   | A | FP4 (release) | FP8 | release |
   | B | Mode B | Mode B | Mode B |
   | C-quality | TQ1 k = 2 + GPTQ | FP8 **and** TQ1 k = 2 | release |
   | C-size | TQ1 k = 1 + GPTQ | FP8 **and** TQ1 k = 1 | release |
   | D | QAT ternary k = 1 | per the approved run | release |

   Mode D on the full model exists only if Vedant has approved and the run has happened. Otherwise
   deliver: the training code at full scale (FSDP + expert parallel), the compute estimate in
   tokens/FLOPs/GPU-hours/cost, the launch scripts, and the M4 evidence for what Mode D is likely to
   cost in quality. **Do not launch full-size QAT without Vedant's explicit confirmation.**
4. **Runtime for the full model.** GPU path for every mode (the CUDA kernels from M3/M4), multi-GPU
   or offload as needed, explicit cold-expert cache (GPU ↔ CPU ↔ disk) with hit rates logged.
   Validate Boilerplate Mode A against the official `inference/` code on GPU on the full model:
   per-layer hidden-state cosine on a few hundred tokens.
5. **OpenAI-compatible HTTP server** with streaming (`/v1/completions`, `/v1/chat/completions`),
   exposed as `boilerplate serve`, plus an lm-eval-harness adapter (an in-process `LM` subclass is
   fine) that supports loglikelihood and generation tasks.
6. **Evaluation, every variant against A:**
   - per-layer hidden-state cosine vs the reference;
   - WikiText-2 test and a fixed C4 validation subset perplexity, fixed context, same tokenizer;
   - lm-eval-harness: MMLU, GSM8K, HumanEval (or the closest runnable equivalents; name them);
   - routing drift: per-layer top-k overlap with Mode A, expert load entropy;
   - file bytes by `stat`, effective b/w per role and whole file, peak RAM/VRAM, prefill and
     decode tok/s, all with the hardware named.
7. **Final report** `docs/reports/M5.md` plus an updated `README.md`: one results table per
   variant, both engram variants side by side, the Mode D gap stated honestly (or "not run" with the
   reason), the list of everything not run, and the measured-vs-projected sizes reconciled.

### M5 acceptance criteria

- `pytest` green, including converter resume and checksum tests and server streaming tests.
- **Mode B is identical to Mode A** on the full model: bit-identical logits on at least 1,024
  tokens.
- **Mode C k = 2 is "close to baseline"**, against a threshold fixed *before* running the evals.
  Proposed default, to be confirmed by Vedant before the eval: WikiText-2 perplexity within +3%
  relative of Mode A and the mean of MMLU/GSM8K/HumanEval within 1.5 points. Report the result
  whether it passes or fails.
- **Mode D**: the gap to Mode A reported on every metric, with no cherry-picking, or the report
  states it was not run and why.
- Every size in the report is a measured file size; every speed and memory number names its
  hardware; every projection is labelled.
- Both engram variants reported for every Mode C variant.

## 8. When to stop and ask

Stop, post a short summary and wait when you reach any gate in section 2, and also when:

- a result contradicts something in `bit_accounting.md` or a milestone report (re-measure first,
  then report the contradiction; do not quietly edit the old numbers);
- a test you did not write fails and the fix would change measured behaviour;
- the reference and the engine disagree beyond tolerance and the cause is not found within a
  reasonable effort.

When you stop, say exactly what is ready, what is blocked, what you need (hardware, budget, a
decision), and what it will cost.

## 9. Deliverables checklist

- [ ] M3: bounded Mode B decode, GPU decode kernel, mixed B + C policy, tests, updated `M3.md`, fixed README
- [ ] M4: standard-decoder path, CUDA ternary + grouped kernels, NEON (or documented blocker), baselines, PTQ arms, QAT module + tests, shakedown (after approval), OLMoE plan/scripts (run after approval), speed comparison, `M4.md` with verdict
- [ ] M5: streaming converter, full-scale GPTQ, all variants × both engram choices (after approval for hardware), GPU runtime + expert cache, `boilerplate serve`, lm-eval adapter, full evals, `M5.md` final report, README
- [ ] Each milestone: PR(s) with green tests and a report

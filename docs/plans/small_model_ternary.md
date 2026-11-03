# Plan: a real ternary checkpoint of a complete small model

Status: proposed 2026-11-03. Not started. It needs a GPU budget decision before the training phase.

## Why

M0–M1 built and validated infrastructure. Nothing so far shows that a ternary model keeps useful
quality at a size and speed that beat conventional quantization. This experiment tests that claim
end to end on a complete pretrained model small enough to train on rented hardware, before more
work goes into the 760 B-parameter target. It replaces the old M4 ("QAT validated on a small model")
and adds a conventional quantized baseline and a published size/quality/speed comparison.

Recommended placement: directly after M2 (it needs TQ1/TQ2 packing and the ternary kernels),
before M3 (lossless Mode B, which `deepseek_v41_flash.md` expects to give small gains on FP4/FP8).

## Models

| Stage | Model | Why |
|---|---|---|
| Shakedown | `HuggingFaceTB/SmolLM2-360M` (dense, BF16) | Cheap: the whole pipeline (convert, QAT, export, engine, eval) runs on one GPU in a few hours. Open pretraining data |
| Main | `allenai/OLMoE-1B-7B-0125` (MoE, 64 experts top-8, ~1.3 B active, 6.9 B total, BF16) | An MoE like the target, BF16 source so ternary saves a lot, fully open pretraining data for QAT |
| Context only | `microsoft/bitnet-b1.58-2B-4T` | A natively trained ternary model: shows what ternary from scratch reaches at this scale |

The engine needs a small "standard decoder" path (RMSNorm, GQA with QK-norm, SwiGLU MoE) to run these
models; the V4.1-specific parts are not needed.

## Arms (same model, same tokenizer, same eval)

1. **BF16** original.
2. **Conventional baselines**: llama.cpp `Q8_0`, `Q4_K_M`, `Q3_K_M`, `Q2_K`, and `IQ2_XS` (with an
   importance matrix). These are the comparisons a reader will ask for.
3. **Post-training ternary, no training**: absmean ternary (TQ1 and TQ2), and Mode C TQ1 k = 2. Expected to
   lose a lot; it shows how much the training buys.
4. **QAT ternary** (the actual claim): BitNet b1.58-style straight-through QAT on all linear layers in the
   blocks (experts, attention, MLP); embeddings, LM head, norms and router stay BF16. Distillation from the
   BF16 model (logit KL + CE). Checkpoints at 1 B, 3 B and 10 B tokens to get a quality-vs-tokens curve.
   Exported to TQ1 and TQ2, and also to llama.cpp `TQ1_0`/`TQ2_0` so our kernels can be compared with
   theirs on identical weights.

## Measurements (all reported per arm)

| What | How |
|---|---|
| File bytes | `stat` of the exported file(s), not a projection; also bits per parameter |
| Peak memory | peak RSS (CPU) and peak allocated VRAM (GPU) during a 2K-token prompt + 256-token decode |
| Speed | decode tok/s at batch 1 and 8, prefill tok/s, same machine per comparison: one 4-core CPU box and one GPU. Boilerplate engine and llama.cpp both reported |
| Perplexity | WikiText-2 test and a held-out slice of the pretraining mix, fixed 2048 context, same tokenizer across arms |
| Tasks | lm-evaluation-harness: MMLU (5-shot), ARC-Challenge, HellaSwag, PIQA, WinoGrande, GSM8K (8-shot). GSM8K is noisy at this size; it is reported, not used for the verdict |
| MoE health | per-layer router top-k overlap with BF16, expert load entropy |

## Success criterion (fixed before running)

The idea works at this scale if the QAT ternary arm either

* beats the best conventional arm of equal or larger file size (expected: `Q2_K` / `IQ2_XS`) on
  both perplexity and the mean of the non-GSM8K task scores, or
* stays within 1 point of `Q4_K_M`'s mean task score at 60% or less of its file bytes,

and its decode speed on the same CPU is at least that of `Q4_K_M`. If neither holds at 10 B tokens,
we report that and revisit the plan before any work on the full model.

## Compute (estimates, not quotes)

* Shakedown, SmolLM2-360M, 2 B tokens with distillation: ~5e18 FLOPs, a few GPU-hours on one 24–80 GB GPU.
* OLMoE, 10 B tokens with a BF16 teacher: ~1e20 FLOPs, roughly 100–200 H100-hours. Optimizer state for
  6.9 B parameters plus the teacher is ~120 GB, so 2–4 × 80 GB GPUs with FSDP.
* Neither fits this cloud container (no GPU) or ZeroGPU (time-limited calls). Baselines, PTQ arms and
  all evaluation except the GPU speed numbers can run on CPU.

## Steps

1. Baselines: download both models, produce the llama.cpp quants, measure bytes, memory, speed, perplexity,
   tasks. CPU only, can start any time.
2. After M2: standard-decoder path in the engine; PTQ ternary arms; kernel comparison vs llama.cpp `TQ*_0`.
3. QAT shakedown on SmolLM2-360M (GPU, small).
4. QAT on OLMoE at 1 B / 3 B / 10 B tokens (GPU budget needed; launched only with Vedant's go-ahead).
5. Report `docs/reports/small_model_ternary.md` with every arm and the verdict against the criterion above.

# Bit accounting: what ternary can and cannot save on this checkpoint

Written 2026-11-03 in response to an external review of the plan. Everything here is arithmetic
from the checkpoint's verified storage formats (`deepseek_v41_flash.md`), not a measurement.

## Per-term cost of TQ1 / TQ2

| Format | Payload | Scale | Bits per weight per term |
|---|---|---|---|
| TQ1 | 5 trits per byte = 1.6 b | FP16 per 128 = 0.125 b | **1.725** |
| TQ2 | 2 b | FP16 per 128 = 0.125 b | **2.125** |
| TQ1, ue8m0 scale per 128 | 1.6 b | 0.0625 b | 1.6625 |
| llama.cpp `TQ1_0` (for comparison) | 256-weight blocks | FP16 per 256 | 1.6875 |
| llama.cpp `TQ2_0` (for comparison) | 256-weight blocks | FP16 per 256 | 2.0625 |

Excludes padding and per-tensor metadata, which are negligible for these tensor sizes.

## Mode C (stacked residual ternary) against the source formats

A stack of `k` ternary terms costs `k` times the per-term bits. Each term needs its own scales,
so stacking never keeps the cost of one ternary matrix.

| k | TQ1 (1.725) | TQ2 (2.125) |
|---|---|---|
| 1 | 1.725 | 2.125 |
| 2 | 3.450 | 4.250 |
| 3 | 5.175 | 6.375 |
| 4 | 6.900 | 8.500 |

Source formats in this checkpoint:

| Role | Source bits/weight | Share of file | Mode C settings that are actually smaller |
|---|---|---|---|
| Routed experts | **4.25** (FP4 e2m1 + ue8m0 per 32) | ~58% | TQ1 k = 1 (0.41×), TQ1 k = 2 (0.81×), TQ2 k = 1 (0.50×). **TQ2 k = 2 is the same size; TQ1 k ≥ 3 and TQ2 k ≥ 3 are larger** |
| Engram tables | 8.25 (FP8 + ue8m0 per 32) | ~40% | everything up to TQ1 k = 4 / TQ2 k = 3 |
| Attention, shared experts, engram projections | ~8.01 (FP8, 32×32 blocks) | ~2% | everything up to TQ1 k = 4 / TQ2 k = 3 |

The project brief's "near-baseline at k = 2–3" assumed 8–16-bit sources. For the routed experts,
which hold most of the non-engram bytes, only k = 1 gives a large saving and TQ1 k = 2 saves 19%.

### Policy rule (applies from M2)

* The precision policy computes each role's effective bits per weight for the chosen format and
  `k`, and refuses a setting whose effective bits are not below `0.85 ×` the source bits for that
  role (override flag for experiments only). For routed experts this allows TQ1 k ≤ 2 and TQ2 k = 1.
* Mode C defaults for routed experts: **TQ1 k = 1** (size) or **TQ1 k = 2** (quality). Not TQ2 k = 2.
* Every Mode C report states the effective bits per weight next to the source bits, per role and
  for the whole file, and compares quality against a conventional quantizer at the same bits where
  one exists (for experts, the FP4 source itself is that baseline).

## Whole-model ceiling

From `deepseek_v41_flash.md`: ~510 GB as shipped. TQ1 routed experts (backbone + MTP) give
~334 GB (1.5×); TQ2 ~1.4×; TQ1 experts plus ternary engram tables ~175 GB (2.9×). Over the ~760 B
stored parameters that is about 3.5 bits per parameter on average when the engram tables keep
FP8, and about 1.8 when they are ternarized too. "1.6-bit DeepSeek" describes the experts; it
describes the whole file only in the variant that also ternarizes the engram tables, whose
quality cost is not yet known.

### Per-token weight traffic and offloading

Each routed expert is ~35.4 M parameters (543.6 B / 40 layers / 384). Top-6 routing over 40 layers
touches 240 experts, ~8.5 B parameters per token:

| Expert format | Expert bytes read per token (all cold) |
|---|---|
| FP4 (source) | ~4.5 GB |
| TQ2 | ~2.3 GB |
| TQ1 | ~1.8 GB |

The TQ1 backbone experts alone are ~117 GB, so on a consumer machine (for example 8 GB VRAM and
32 GB RAM) most of them stay on disk. If three quarters of expert reads miss RAM, a TQ1 decode
step reads ~1.4 GB from disk: about 0.4 s per token at 3.5 GB/s NVMe, before any compute. Routing
locality and expert caching can improve this, and that has to be measured. Ternary makes offloaded
decoding about 2.5× cheaper in bytes than FP4; it does not make the model resident.

## Prior work and where a contribution could be

Ternary LLMs are not new: BitNet b1.58 trains ternary models from scratch and has released
ternary checkpoints, and llama.cpp already ships `TQ1_0` and `TQ2_0` with CPU kernels. Ternary
weights by themselves are therefore not the contribution. Candidate contributions, each of which
needs its own evidence:

1. Conversion: QAT/distillation recipes that bring a pretrained FP4/FP8 MoE to ternary at a
   measured quality cost, with expert-routing drift measured and controlled.
2. Precision allocation: which roles of a modern MoE (experts, engram tables, hyper-connections,
   attention) tolerate ternary, measured per role.
3. Kernels: multiply-free ternary GEMV/GEMM that beat the best FP4 / low-bit kernels on the same
   hardware at equal quality.
4. Deployment: offloaded MoE decoding where ternary experts change what fits and how fast it runs.

The first experiment that tests the central idea end to end is in
`docs/plans/small_model_ternary.md`.

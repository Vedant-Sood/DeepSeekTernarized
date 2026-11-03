# DeepSeek V4.1 Flash 1017: what the checkpoint actually contains

Source: `deepseek-ai/DeepSeek-V4.1-Flash-1017` on the Hugging Face Hub, read on 2026-11-03:
`config.json`, `model.safetensors.index.json` (names only), the shard file list with sizes, and
the official reference code in `inference/` (`model.py`, `kernel.py`, `convert.py`, `config.json`).
Shard *headers* could not be read from the build container (see the M0 report), so dtypes below
come from the reference code and `quantization_config`, not from headers. `boilerplate inspect
deepseek-ai/DeepSeek-V4.1-Flash-1017` reads the headers over HTTP range requests and verifies them.

## Architecture (from config.json, interpreted with inference/model.py)

| | |
|---|---|
| Class | `DeepseekV41ForCausalLM`, `model_type: deepseek_v41`, text config nested under `text_config`, plus a 32-layer ViT (`vision_config`) |
| Backbone | 40 layers, hidden 5120, vocab 129,280, RMSNorm eps 1e-20, no tied embeddings |
| Residual stream | Hyper-Connections: 4 parallel copies (`hc_mult`), Sinkhorn-normalized mixing (20 iters) |
| Attention | Not MLA. 64 query heads with LoRA'd queries (rank 1280), a single shared 512-d KV head (`num_key_value_heads: 1`), 64 RoPE dims, grouped low-rank output (8 groups, rank 1024), per-head attention-sink logits |
| Attention sparsity | Every layer has a 128-token sliding window. Layers 2–19 add 2:1 compressed KV, layers 20–39 add 1:1 compressed KV; layers 0–1 are window-only. Compressed KV is produced only by layers 2, 8, 14, 20 and shared by the following layers; a top-512 indexer runs at layers 2, 8, 14, 20, 24, 28, 32, 36, with two-level candidate-block prefiltering from layer 20 |
| RoPE | Window-only layers: plain RoPE, θ = 10,000. Compressed layers: YaRN (factor 16, original 65,536) at θ = 160,000 |
| MoE | Every layer: 384 routed experts + 1 shared, top-6, `sqrtsoftplus` scores, selection-only correction bias (separate bias for image tokens), normalized top-k weights × 1.5, SwiGLU with clamp 10 |
| Engram | n-gram hash lookup tables injected at layers 1 and 14: 384,006,168 and 384,016,682 rows × 256 dims |
| MTP / DSpark | 3 draft stages under `mtp.*`, 128 experts top-3 each, 5-token draft blocks |

## Storage

`quantization_config`: `fp8`, dynamic activations, `weight_block_size [32, 32]`, `scale_fmt ue8m0`,
`expert_dtype fp4`. From the reference code:

* Routed experts: **FP4 e2m1**, packed two per byte as int8 `[out, in/2]`, one ue8m0 scale per 32 inputs (4.25 bits/weight).
* Other large linears (attention, shared experts, engram projections): FP8 e4m3 with 32×32 ue8m0 block scales (~8.01 bits/weight).
* Engram tables: FP8 e4m3, one ue8m0 scale per 32 values (8.25 bits/weight).
* Embeddings, LM head, small projections: BF16. Hyper-connection params, attention sinks, router biases: FP32.

## Where the bytes are (arithmetic from config + file sizes, not a measurement)

Index `total_size`: 510.3 GB.

| Part | Params | Bytes | Notes |
|---|---|---|---|
| Routed experts (backbone) | 543.6 B | ~288.8 GB | FP4 + scales; shards 3–42 are ~7.39 GB each, matching |
| Routed experts (MTP stages) | 13.6 B | ~7.2 GB | |
| Engram tables | 196.6 B | ~202.8 GB | the two ~101.5 GB shards (47, 48) |
| Everything else | | ~11.5 GB | attention, shared experts, embeddings, head, vision, ... |

### Consequence for the size goal

The project brief assumed 8–16-bit source weights, hence "5–8× smaller". This checkpoint's experts
are already 4.25-bit. Replacing the routed experts with TQ1 (1.6 bits + FP16 scale per 128 = 1.725
bits/weight) would take them from ~296 GB to ~120 GB, and the whole checkpoint from ~510 GB to
~334 GB (≈1.5×). TQ2 gives ≈1.4×. Ternarizing the engram tables too would bring the total to
roughly 175 GB (≈2.9×). These are projections; the real numbers get measured at M2/M5.

Runtime memory is a different story: the engram tables are lookups (a few rows per token) and can
stay mmap'd on disk, so the GPU-resident weight footprint is dominated by the experts, roughly
300 GB at FP4 versus roughly 130 GB with TQ1 experts.

Mode B (lossless entropy coding) targets low-entropy exponent bits. FP4 e2m1 has 2 exponent bits
and FP8 e4m3 has 4, so the DFloat11-style gains seen on BF16 should not be expected here. M3 will
measure it.

## Reference runtime

HF transformers (5.22) implements `deepseek_v4` but not `deepseek_v41`, and the two differ (engram,
shared compressed KV and indexers, compression ratios, vision). The trusted reference is therefore
the official `inference/model.py`, which needs TileLang GPU kernels. Boilerplate ships a pure-PyTorch
drop-in for its `kernel` module (`boilerplate/reference/cpu_kernels.py`), so the unmodified
reference code can run on CPU for small models.

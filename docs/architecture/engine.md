# Boilerplate engine: Mode A design (M1)

Mode A runs the checkpoint's weights in their stored formats. It is the in-engine reference that
Modes B–D are compared against, so it is validated against the official DeepSeek code first.

## Code map

| File | Role |
|---|---|
| `boilerplate/engine/config.py` | `EngineConfig` from the checkpoint's `config.json`, plus the derived per-layer plan (which layer's compressed KV and indexer each layer reads) |
| `boilerplate/engine/weights.py` | `WeightStore`: every shard memory-mapped copy-on-write; each tensor is a zero-copy view |
| `boilerplate/engine/model.py` | `V4Model.forward`: one step over any mix of prefilling and decoding sequences |
| `boilerplate/engine/kv_cache.py` | `KVCache`: window rings, paged compressed KV and index keys, compressor tails, engram history |
| `boilerplate/engine/engram.py` | n-gram hashing (primes, compressed token map, multipliers, per-sequence history) |
| `boilerplate/engine/engine.py` | Python API: `Engine.load`, `generate`, `generate_text`, `trace` |
| `boilerplate/kernels/torch_ref.py` | Pure-PyTorch kernels with the official TileLang kernels' numerics (FP8/FP4 activation quant, FP8 and FP4 GEMM, sparse attention with sinks, Sinkhorn) |
| `third_party/deepseek_v41_inference/` | The official reference code, vendored unmodified (MIT) so equivalence tests run offline |

## Step layout

A step takes `(sequence, new tokens)` pairs. All tokens are flattened into one `[1, T, ...]` batch,
so the embedding, every projection, the hyper-connection mixing, engram and the MoE run once over
all `T` tokens. Attention is the only per-sequence part: each sequence attends over its own window
ring and its own compressed-KV pages. A step can mix sequences that are prefilling (start 0, any
length) with sequences that are decoding (one token each). Chunked prefill (start > 0 and several
tokens) is not supported yet; the reference does not have it either.

MoE is grouped by expert: router top-k over all tokens, then each selected expert runs once over
all of its tokens, accumulating in FP32 in expert order (the reference's order, which keeps the
sums bit-identical).

## Weights

Weights stay in their stored formats; nothing is upcast at load except where the reference does it:

* FP8 e4m3 linears with ue8m0 32×32 block scales; activations quantized per 32 values with
  power-of-two scales before the GEMM.
* Routed experts: packed FP4 e2m1 with ue8m0 per-32 scales.
* `wo_a` (grouped output LoRA) dequantized to BF16 at load, as the reference's `convert.py` does.
* Compressor projections: FP32 for pooled ratios, BF16 for ratio 1. Router and head computed in FP32.

All shards are mmap'd, so routed experts and engram tables (most of the bytes) are only read when
a token routes to them. Experts are wrapped lazily on first use. This is the baseline cold-expert
offload: the OS page cache decides what stays in RAM. An explicit GPU/CPU expert cache comes with
the GPU kernels.

## KV cache

V4.x has no per-head KV: each layer has one 512-d latent per token that serves as both K and V.

| State | Owner | Shape per sequence | Storage |
|---|---|---|---|
| Window KV | every layer | `window_size × head_dim` | fixed ring, slot = position % window |
| Compressed KV | KV-source layers (others read the source's) | `len / ratio × head_dim` | **paged** |
| Index keys | KV-source layers with an indexer | `len / ratio × index_head_dim` | **paged** |
| Compressor tail | ratio > 1 sources | `ratio × head_dim` (×2, FP32) | per-sequence slot |
| Engram history | per sequence | `max_ngram_size − 1` ids | per sequence |

Paging uses one page table per sequence, indexed by token position: logical page `p` covers tokens
`[p·P, (p+1)·P)`, so for a ratio-`r` source a physical page holds `P / r` compressed vectors and one
table serves every pool. `P` (default 64) must be a multiple of every compress ratio. Pages are
allocated as a sequence grows and returned when it is freed. Values are stored in BF16 after the
same FP8/FP4 rounding the reference applies, so a Mode A cache holds exactly the reference's
values; storing the narrow codes instead (about 3.5× less cache memory) is a later optimization.

For V4.1 Flash the KV sources are layers 2, 8, 14 (ratio 2) and 20 (ratio 1), each with index
keys. Per token that is 1024 B × (3 × ½ + 1) of compressed KV plus 256 B × (3 × ½ + 1) of index
keys ≈ 3.2 KB, plus 40 fixed windows of 128 × 1 KB ≈ 5.2 MB per sequence. A 64K-token context is
therefore about 210 MB of cache. (Arithmetic from the config, not a measurement.)

## Kernels

M1 uses one backend, `boilerplate.kernels.torch_ref`, which reproduces the official TileLang
kernels' numerics in plain PyTorch. That is what lets the engine be compared bit for bit with the
reference running on the same kernels. It is a correctness backend: it dequantizes each weight per
call and is not fast. The multiply-free ternary kernels (M2), the entropy-decode kernel (M3) and
CUDA/AVX/NEON paths plug in behind the same `QLinear` interface.

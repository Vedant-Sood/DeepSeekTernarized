# Boilerplate checkpoint format (version 1)

A Boilerplate checkpoint is a directory:

```
config.json                       copied from the source checkpoint
tokenizer.json, tokenizer_config.json
manifest.json                     what every tensor is and how it is stored
model.safetensors.index.json      stored tensor name -> shard (standard HF index)
model-00001-of-0000N.safetensors  shards (default 2 GiB each)
```

`boilerplate export SRC OUT --policy configs/policy/<name>.yaml` writes one. The engine
(`Engine.load(OUT)`) reads it, and so does any directory that has no manifest (that is a plain
source checkpoint, so every tensor is Mode A).

## Policy

A YAML list of rules, tried in order. The first rule whose regex `search`es a weight's base name
(the tensor name without `.weight`) decides how that weight is stored. Tensors that are not weights
(norms, biases, hyper-connection parameters, attention sinks) are always Mode A.

```yaml
name: experts-k2
default: {mode: A}
rules:
  - match: '^layers\.\d+\.ffn\.experts\.\d+\.w[123]$'
    mode: C          # A original | B lossless entropy-coded | C stacked ternary
    k: 2             # ternary terms, 1..4
    group: 128       # input columns per scale
    packing: tq1     # tq1 (1.6 b/w) | tq2 (2 b/w)
    scale_dtype: fp16
    act: int8        # int8 per-token absmax activations into the layer | bf16
    gptq: false      # GPTQ error compensation (needs --calib-text at export)
    allow_larger: false
```

A Mode C setting costs `k * (bits(packing) + 16 / group)` bits per weight. The exporter compares
that with what the source tensor actually uses, including its scales, and refuses a setting that
is not below 0.85x the source unless `allow_larger: true`. V4.1's routed experts are FP4 with one
e8m0 scale per 32 weights, so they use 4.25 b/w. TQ2 k=2 costs 4.25 b/w too, so it is refused.
TQ1 k=2 costs 3.45 b/w and is allowed.

Shipped policies (`configs/policy/`):

| file | what changes | Mode C bits/weight |
|---|---|---|
| `experts_k2.yaml` | routed experts, TQ1 k=2 | 3.45 |
| `experts_k1.yaml` | routed experts, TQ1 k=1 | 1.725 |
| `experts_engram_k2.yaml` | routed experts and both engram tables, TQ1 k=2 | 3.45 |
| `lossless.yaml` | every weight Mode B | n/a |
| `experts_k2_lossless_rest.yaml` | routed experts Mode C; other supported weights Mode B | 3.45 for experts |

## Stored tensors per mode

For a weight with base name `X`:

* **Mode A**: `X.weight` (and `X.scale`), byte for byte as in the source.
* **Mode B**: each source tensor `T` (`X.weight`, `X.scale`) with 1- or 2-byte elements becomes
  - `T.hstream` uint8: the Huffman payload, followed by 8 zero bytes of padding;
  - `T.hoffsets` int64 `[blocks + 1]`: the bit where each block starts, plus the total bit count;
  - `T.hlens` uint8 `[256]`: the canonical code length of each byte value (0 = unused);
  - `T.hraw` uint8 (2-byte dtypes only): the low byte of each element, uncoded.

  The coded symbols are every byte for 1-byte dtypes (FP8, packed FP4, e8m0 scales), and the
  high byte (sign, exponent and top mantissa bit) for BF16/FP16. Codes are canonical and
  MSB-first, at most 12 bits long, and come in blocks of 4096 symbols. Each block decodes
  independently with one 4096-entry table of `(symbol << 8) | length`, indexed by the next 12 bits.
  4-byte tensors are stored as in Mode A.

  The CPU engine decodes coded linears on demand. Embeddings and engram tables are decoded by
  the independent Huffman blocks covering requested rows; the language-model head is decoded
  in row chunks. The decoded-byte LRU is controlled by `mode_b_cache_bytes` in `Engine.load` or
  `--mode-b-cache-mib` in the CLI. It bounds retained cache entries, not temporary decode
  buffers, PyTorch's separate FP32 dequantization cache, file-backed mmap pages, or process RSS.
* **Mode C**: for each term `i` in `0..k-1`:
  - `X.tq{i}.packed` uint8 `[N, ceil(K / per_byte)]`: the trits, row-major, with the reduction dim
    K contiguous;
  - `X.tq{i}.scale` fp16/bf16 `[N, ceil(K / group)]`.

  The weight is `W[n, j] = sum_i scale_i[n, j // group] * trit_i[n, j]`.

### Trit packing

Each trit `t` is stored as the digit `u = t + 1`.

* **TQ2**: `byte = u0 | u1 << 2 | u2 << 4 | u3 << 6` (the first weight goes in the low bits).
* **TQ1**: `byte = u0 + 3 u1 + 9 u2 + 27 u3 + 81 u4` (the first weight is the least significant
  digit, and the byte is at most 242).

Rows are padded with the digit 1 (t = 0) up to a whole byte. Both kernels decode one packed row to
int8 trits once and reuse it for every activation row. The CPU kernel needs no particular group
alignment, but groups of 128 or 256 keep each group's inner loop a whole number of 32-byte vectors.

## manifest.json

```jsonc
{
  "format": "boilerplate", "version": 1,
  "source": "...", "architecture": { ...source config.json... },
  "policy": { "name": ..., "rules": [...] },
  "summary": {
    "params": ..., "bytes": ..., "bits_per_weight": ...,          // whole model, everything stored
    "by_mode": {"A": {"params", "bytes", "bits_per_weight"}, "C": {...}, "B": {...}}
  },
  "tensors": {
    "layers.0.ffn.experts.0.w1": {
      "mode": "C", "shape": [2304, 5120], "source_dtype": "fp4_e2m1", "source_bits": 4.25,
      "packing": "tq1", "terms": 2, "group": 128, "scale_dtype": "fp16", "act": "int8", "gptq": false,
      "stored": {"layers.0.ffn.experts.0.w1.tq0.packed": {"dtype": "uint8", "shape": [2304, 1024],
                 "sha256": "...", "file": "model-00002-of-00003.safetensors"}, ...},
      "bytes": 5087232, "params": 11796480
    },
    "layers.0.attn.wq_a": {"mode": "B", "coded": {"layers.0.attn.wq_a.weight": {"dtype": "float8_e4m3fn",
                            "shape": [...], "symbols": ...}}, "maxlen": 12, "block": 4096, ...}
  }
}
```

`params` counts logical weights: a packed FP4 byte holds 2 of them, and scales are not counted.
`bytes` counts everything stored for the weight, scales included. So `summary.bits_per_weight` is
the effective bits per weight over the whole file. Every stored tensor has a SHA-256 checksum of
its bytes.

# Measurement scripts behind docs/reports/M2.md and M3.md

These are the scripts as they were run, with the paths used in the development container
(`/home/user/work/...`). Change the paths to rerun them.

| script | report section |
|---|---|
| `m2_eval.py BASE CAND1,CAND2 IDS.safetensors N` | Mode C / Mode B vs Mode A: per-layer cosine, logit KL, top-1, NLL |
| `gptq_eval.py LAYER NEXPERTS` | GPTQ vs plain fit, output error on held-out real expert inputs |
| `engram_sample.py`, `engram_eval.py` | real engram rows via HTTP range requests; ternary error and Mode B gain |
| `modeb_measure.py CKPT` | Mode B on every tensor: exact round trip, ratio per role/dtype, codec throughput |
| `modeb_validate.py SOURCE MODE_B` | Exact Mode A/Mode B trace bytes over a prefill and decode, with configurable decoded-cache budgets |
| `reference_trace_rss.py BACKEND CKPT IDS TRACE` | Measure official-reference or Mode A load and trace peak RSS in a separate process |
| `compare_trace_bytes.py REF CAND` | Compare input, layer and logit bytes from saved traces |
| `fullsize.py` | per-role params and bytes of the full release from its shard headers |

For a **local 7-layer slice** and a fresh output directory, activate the project virtualenv and
ensure the PyTorch extension cache is writable (in the cloud environment,
`export TORCH_EXTENSIONS_DIR=/workspace/.cache/torch_extensions`). Then:

```bash
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MKL_DYNAMIC=FALSE
test ! -e "$SLICE_MODE_B" && boilerplate export "$SLICE" "$SLICE_MODE_B" --policy configs/policy/lossless.yaml
boilerplate bench "$SLICE" --batch 1 --prompt-len 256 --gen-len 16 --threads 4 --mode-b-cache-mib 256
boilerplate bench "$SLICE_MODE_B" --batch 1 --prompt-len 256 --gen-len 16 --threads 4 --mode-b-cache-mib 256
python scripts/eval/modeb_validate.py "$SLICE" "$SLICE_MODE_B" --tokens 256 --decode-from 248 --threads 4 --budgets-mib 0 256
```

`modeb_validate.py` exits nonzero unless every layer and logits have identical bytes for each
budget. It generates seeded token IDs by default; `--input-ids FILE.safetensors` uses the file's
`input_ids` tensor instead. It does not measure the process peak RSS; use the separate `bench`
processes above for that comparison. Real-slice results and exact workload details are in the
2026-11-04 follow-up in `docs/reports/M3.md`.

# Measurement scripts behind the milestone reports

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
| `standard_perplexity.py CKPT IDS --state FILE` | Score fixed token IDs in 2,048-token windows, saving resumable progress after every window |
| `standard_split_trace.py record ...` / `compare ...` | Save one standard-decoder prompt at a time in separate reference/engine processes and compare their checksummed traces |
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

For standard-decoder perplexity, `IDS` is a one-dimensional safetensors `input_ids` tensor.
The script scores the same last-half positions in each complete context window as the pinned
llama.cpp default. It records checkpoint and input hashes, hardware and thread settings. A
partial run can continue with the same command plus `--resume`; a changed input, checkpoint or
measurement environment is rejected. The first-window diagnostics and file hashes are in
`docs/reports/small_model_ternary.md`.

The OLMoE reference and engine can be compared in separate processes, so the two models do not
need to be resident together. With local checkpoint and prompt-ID paths set in `OLMOE` and
`IDS`, and new output directories set in `HF_TRACE` and `ENGINE_TRACE`:

```bash
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MKL_DYNAMIC=FALSE OMP_DYNAMIC=FALSE
python scripts/eval/standard_split_trace.py record hf "$OLMOE" "$IDS" "$HF_TRACE" --tokens 64 --threads 4 --count 20
python scripts/eval/standard_split_trace.py record engine "$OLMOE" "$IDS" "$ENGINE_TRACE" --tokens 64 --threads 4 --count 20
python scripts/eval/standard_split_trace.py compare "$HF_TRACE" "$ENGINE_TRACE" --threads 4 --out comparison.json
```

Add `--resume` to either record command to keep completed prompts after an interruption. The
native FP32 Hugging Face OLMoE load and two-token smoke test peaked at 31.7760 GiB RSS on the
32 GiB CPU container; the full comparison has not been run.

# Measurement scripts behind docs/reports/M2.md and M3.md

These are the scripts as they were run, with the paths used in the development container
(`/home/user/work/...`). Change the paths to rerun them.

| script | report section |
|---|---|
| `m2_eval.py BASE CAND1,CAND2 IDS.safetensors N` | Mode C / Mode B vs Mode A: per-layer cosine, logit KL, top-1, NLL |
| `gptq_eval.py LAYER NEXPERTS` | GPTQ vs plain fit, output error on held-out real expert inputs |
| `engram_sample.py`, `engram_eval.py` | real engram rows via HTTP range requests; ternary error and Mode B gain |
| `modeb_measure.py CKPT` | Mode B on every tensor: exact round trip, ratio per role/dtype, codec throughput |
| `fullsize.py` | per-role params and bytes of the full release from its shard headers |

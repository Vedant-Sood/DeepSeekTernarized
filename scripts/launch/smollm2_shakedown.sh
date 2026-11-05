#!/usr/bin/env bash
# SmolLM2-360M ternary QAT shakedown on one GPU (Vedant's RTX 4060 laptop).
# GATED: the trainer refuses more than 20 steps until configs/qat/smollm2_shakedown.yaml has
# `approval:` filled in after Vedant approves the GPU budget. Steps 1-4 need no approval.
#
#   scripts/launch/smollm2_shakedown.sh prepare   # download, tokenize, teacher top-k
#   scripts/launch/smollm2_shakedown.sh probe     # 20-step throughput probe, then re-estimate
#   scripts/launch/smollm2_shakedown.sh train     # the approved run (resumes automatically)
#   scripts/launch/smollm2_shakedown.sh eval      # perplexity + lm-eval on the Mode D exports
set -euo pipefail
CFG=${CFG:-configs/qat/smollm2_shakedown.yaml}
MODEL=work/models/smollm2-360m
DATA=work/data
TEACHER=work/qat/smollm2_teacher_top32
TRAIN_TEXT=${TRAIN_TEXT:-$DATA/fineweb_edu_sample.txt}   # a UTF-8 text sample of the pretraining mix

case "${1:-}" in
  prepare)
    huggingface-cli download HuggingFaceTB/SmolLM2-360M --local-dir "$MODEL"
    boilerplate qat-tokenize "$TRAIN_TEXT" "$MODEL/tokenizer.json" "$DATA/fineweb_edu_smollm2.bin"
    boilerplate qat-teacher "$MODEL" "$DATA/fineweb_edu_smollm2.bin" "$TEACHER" --seq-len 2048 --k 32
    boilerplate qat-estimate "$CFG" --gpu rtx4060-laptop
    ;;
  probe)
    boilerplate qat-train "$CFG" --max-steps 20
    python - "$CFG" <<'PY'
import json, sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
rows = [json.loads(l) for l in open(f"{cfg['out_dir']}/log.rank0.jsonl") if '"tok_s"' in l]
print(f"measured tok/s (last log window): {rows[-1]['tok_s']:.0f}; peak VRAM: {rows[-1].get('max_vram_bytes', 0) / 2**30:.2f} GiB")
print("re-estimate:  boilerplate qat-estimate", sys.argv[1], "--gpu rtx4060-laptop --tok-s", round(rows[-1]["tok_s"]))
PY
    echo "Probe done. Delete $(python -c "import yaml;print(yaml.safe_load(open('$CFG'))['out_dir'])") before the real run so it starts from step 0."
    ;;
  train)
    boilerplate qat-train "$CFG"
    ;;
  eval)
    OUT=$(python -c "import yaml;print(yaml.safe_load(open('$CFG'))['out_dir'])")
    for d in "$OUT"/mode_d_*; do
      python scripts/eval/standard_ppl.py "$d" "$DATA/wikitext2_test.txt" --out "$d/ppl_wikitext2.json"
      boilerplate lm-eval "$d" --tasks arc_easy,hellaswag,piqa,winogrande --out "$d/lm_eval.json" --max-seq-len 2048
    done
    ;;
  *) sed -n 2,10p "$0"; exit 2 ;;
esac

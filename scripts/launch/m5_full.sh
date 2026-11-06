#!/usr/bin/env bash
# M5: full DeepSeek V4.1 Flash conversion, evaluation and serving, stage by stage.
#
# GATED stages (download, convert, gptq, eval on the full model) refuse to run unless
# APPROVED_BY="<who, date, hardware budget>" is set: Vedant approves the hardware first. Mode D on
# the full model is not here at all: full-size QAT needs Vedant's explicit confirmation and a
# trainable V4.1 model (see docs/reports/M5.md, "not built").
#
#   scripts/launch/m5_full.sh plan                 # disk/RAM/time needs, no work done
#   APPROVED_BY=... scripts/launch/m5_full.sh download
#   APPROVED_BY=... scripts/launch/m5_full.sh convert   # Variant B + both engram choices of C (resumable)
#   APPROVED_BY=... scripts/launch/m5_full.sh eval      # every variant vs A: cosine, ppl, lm-eval, size, speed
#   scripts/launch/m5_full.sh serve <variant>      # OpenAI-compatible server on one variant
#
# Hardware the plan assumes (refine before asking): >= 1.6 TB free disk (510 GB source + ~300 GB per
# ternary variant x 4 + Mode B), >= 64 GB RAM for the streaming converter, one or more 80 GB GPUs for
# GPTQ calibration and for the evals (CPU works for everything but takes days at this size).
set -euo pipefail
W=${W:-work/m5}
SRC=$W/source                 # deepseek-ai/DeepSeek-V4.1-Flash-1017
CALIB=${CALIB:-$W/data/calib_c4_512x512.txt}
WIKI=${WIKI:-$W/data/wikitext2_test.txt}
C4=${C4:-$W/data/c4_validation_subset.txt}
VARIANTS="B c_quality c_quality_engram c_size c_size_engram"

gate() { [ -n "${APPROVED_BY:-}" ] || { echo "stage '$1' is gated: set APPROVED_BY after Vedant approves the hardware" >&2; exit 3; }; }
policy() { case "$1" in B) echo configs/policy/lossless.yaml ;; *) echo "configs/policy/m5_$1.yaml" ;; esac; }
dir() { [ "$1" = A ] && echo "$SRC" || echo "$W/variant_$1"; }

case "${1:-}" in
  plan)
    python scripts/eval/fullsize.py 2>/dev/null || echo "(fullsize.py needs work/real_headers.json; see docs/reports/M3.md)"
    echo "variants: A (source) $VARIANTS; each C variant x engram FP8/TQ1 per docs/plans"
    ;;
  download)
    gate download
    huggingface-cli download deepseek-ai/DeepSeek-V4.1-Flash-1017 --local-dir "$SRC"
    ;;
  convert)
    gate convert
    for v in $VARIANTS; do
      extra=()
      grep -q 'gptq: true' "$(policy "$v")" && extra=(--calib-text "$CALIB" --calib-seqs 512 --calib-len 512 --calib-min-tokens 64)
      # resumable: rerunning picks up after the last finished shard (.export_journal.json)
      boilerplate export "$SRC" "$(dir "$v")" --policy "$(policy "$v")" --shard-gb 4 "${extra[@]}"
    done
    ;;
  eval)
    gate eval
    mkdir -p "$W/results"
    boilerplate reference official "$SRC" --out "$W/results/ref" --code-dir "$SRC/inference" \
      --work-dir "$W/ref_work" --gpu-kernels --text-only --seq-len 512 --decode-from 256
    for v in A $VARIANTS; do
      d=$(dir "$v"); r="$W/results/$v"; mkdir -p "$r"
      boilerplate trace "$d" --out "$r/trace" --seq-len 512 --decode-from 256
      boilerplate compare "$W/results/ref" "$r/trace" --min-cos 0.99 > "$r/cosine.txt" || true
      python scripts/eval/standard_ppl.py "$d" "$WIKI" --ctx 2048 --out "$r/ppl_wikitext2.json"
      python scripts/eval/standard_ppl.py "$d" "$C4" --ctx 2048 --out "$r/ppl_c4.json"
      boilerplate lm-eval "$d" --tasks mmlu,gsm8k,humaneval --out "$r/lm_eval.json" --max-seq-len 4096
      boilerplate bench "$d" --prompt-len 512 --gen-len 64 --batch 1 --expert-cache-mib 65536 > "$r/bench_b1.json"
      boilerplate bench "$d" --prompt-len 512 --gen-len 64 --batch 8 --expert-cache-mib 65536 > "$r/bench_b8.json"
      du -b "$d"/*.safetensors | awk '{s+=$1} END {print s}' > "$r/bytes.txt"
    done
    ;;
  serve)
    # Loopback by default. For HOST=0.0.0.0 export BOILERPLATE_API_KEY first (serve refuses otherwise)
    # and put a TLS reverse proxy in front; the key is read from the environment, not the command line.
    boilerplate serve "$(dir "${2:?variant}")" --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}" --max-seq-len 8192
    ;;
  *) sed -n 2,14p "$0"; exit 2 ;;
esac

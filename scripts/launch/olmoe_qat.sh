#!/usr/bin/env bash
# OLMoE-1B-7B ternary QAT with FSDP2 on one node (default 4 GPUs).
# GATED: needs Vedant's approved GPU budget in the config's `approval:` field; the trainer refuses
# runs over 20 steps without it. `prepare` and `dryrun` need no approval.
#
#   scripts/launch/olmoe_qat.sh prepare                      # download F32, cast BF16, tokenize, teacher
#   scripts/launch/olmoe_qat.sh dryrun                       # 2-process CPU FSDP dry run on a tiny OLMoE
#   scripts/launch/olmoe_qat.sh probe configs/qat/olmoe_10b.yaml   # 20 steps, measured tok/s
#   scripts/launch/olmoe_qat.sh train configs/qat/olmoe_10b.yaml   # the approved run (resumes)
set -euo pipefail
NPROC=${NPROC:-4}
SRC=work/models/olmoe-1b-7b-0125
BF16=work/models/olmoe-1b-7b-0125-bf16
DATA=work/data
TRAIN_TEXT=${TRAIN_TEXT:-$DATA/olmoe_mix_sample.txt}

case "${1:-}" in
  prepare)
    huggingface-cli download allenai/OLMoE-1B-7B-0125 --local-dir "$SRC"
    boilerplate cast "$SRC" "$BF16" bf16      # ~28 GB F32 -> ~14 GB BF16, SHA-256 per shard in cast.json
    boilerplate qat-tokenize "$TRAIN_TEXT" "$BF16/tokenizer.json" "$DATA/olmoe_mix_train.bin"
    boilerplate qat-teacher "$BF16" "$DATA/olmoe_mix_train.bin" work/qat/olmoe_teacher_top32 --seq-len 2048 --k 32
    ;;
  dryrun)
    python -m boilerplate.qat.smoke --world 2 --model olmoe
    ;;
  probe)
    torchrun --standalone --nproc-per-node "$NPROC" -m boilerplate.qat.train "$2" --max-steps 20
    ;;
  train)
    torchrun --standalone --nproc-per-node "$NPROC" -m boilerplate.qat.train "$2"
    ;;
  *) sed -n 2,10p "$0"; exit 2 ;;
esac

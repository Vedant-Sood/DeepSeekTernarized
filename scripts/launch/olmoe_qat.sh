#!/usr/bin/env bash
# OLMoE-1B-7B ternary QAT with FSDP2 on one node (default 4 GPUs).
# GATED: needs Vedant's approved GPU budget in the config's `approval:` field; the trainer refuses
# runs over 20 steps without it. `prepare` and `dryrun` need no approval.
#
#   scripts/launch/olmoe_qat.sh prepare                      # download F32, cast BF16, corpus, tokenize
#   scripts/launch/olmoe_qat.sh dryrun                       # 2-process CPU FSDP dry run on a tiny OLMoE
#   scripts/launch/olmoe_qat.sh probe configs/qat/olmoe_10b.yaml   # 20 steps, measured tok/s
#   scripts/launch/olmoe_qat.sh train configs/qat/olmoe_10b.yaml   # the approved run (resumes)
set -euo pipefail
NPROC=${NPROC:-4}
SRC=work/models/olmoe-1b-7b-0125
BF16=work/models/olmoe-1b-7b-0125-bf16
DATA=work/data
CORPUS=$DATA/olmoe_corpus
CHARS=${CHARS:-4.4e10}          # ~11 B tokens of text: enough for the 10 B-token run

case "${1:-}" in
  prepare)
    huggingface-cli download allenai/OLMoE-1B-7B-0125 --local-dir "$SRC"
    boilerplate cast "$SRC" "$BF16" bf16      # ~28 GB F32 -> ~14 GB BF16, SHA-256 per shard in cast.json
    # OLMoE-mix-0924 subsets weighted by their token counts (README); last file of each held out
    M=allenai/OLMoE-mix-0924
    boilerplate qat-data "$CORPUS" --train-chars "$CHARS" --heldout-chars 4e7 \
      --source "$M:data/dclm/*:3860" --source "$M:data/starcoder/*:101" --source "$M:data/pes2o/*:57.2" \
      --source "$M:data/open-web-math/*:12.7" --source "$M:data/algebraic-stack/*:12.6" --source "$M:data/wiki/*:3.69"
    boilerplate qat-tokenize "$CORPUS/train.jsonl" "$BF16/tokenizer.json" "$DATA/olmoe_mix_train.bin"
    boilerplate qat-tokenize "$CORPUS/heldout.jsonl" "$BF16/tokenizer.json" "$DATA/olmoe_mix_heldout.bin"
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

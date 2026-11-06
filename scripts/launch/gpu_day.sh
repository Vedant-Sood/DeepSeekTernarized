#!/usr/bin/env bash
# GPU day, in one command: everything that could not be checked without a GPU, in order, with a
# results folder to send back. Nothing here trains past 20 steps (the trainer's approval gate).
#
#   scripts/launch/gpu_day.sh              # all steps
#   STEP=kernels scripts/launch/gpu_day.sh # one step: env | build | tests | kernels | engine | probe
#
# Needs: an NVIDIA GPU, a CUDA toolkit matching torch.version.cuda (nvcc on PATH or CUDA_HOME set;
# on WSL2 install the "cuda-toolkit" package, not a driver), `pip install -e ".[hf,dev,serve,data]"`.
# Optional: bitsandbytes (8-bit AdamW for the 8 GB probe).
set -uo pipefail
OUT=${OUT:-work/gpu_day/$(date +%Y%m%d-%H%M)}
MODEL=${MODEL:-work/models/smollm2-360m}
mkdir -p "$OUT"
log() { echo "== $*" | tee -a "$OUT/summary.txt"; }
run() { local name=$1; shift; log "$name: $*"; if "$@" > "$OUT/$name.log" 2>&1; then log "$name: OK"; else log "$name: FAILED (see $OUT/$name.log)"; fi; }
want() { [ -z "${STEP:-}" ] || [ "$STEP" = "$1" ]; }

if want env; then
  run env bash -c 'nvidia-smi; python -c "import torch,sys; print(sys.version); print(\"torch\", torch.__version__, \"cuda\", torch.version.cuda, \"available\", torch.cuda.is_available()); print(torch.cuda.get_device_name() if torch.cuda.is_available() else \"no GPU\")"; which nvcc && nvcc --version; echo CUDA_HOME=${CUDA_HOME:-}'
fi
if want build; then
  # JIT-compile both CUDA extensions (Huffman decoder, ternary GEMM); errors are not hidden
  run build python -c "
from boilerplate.kernels.cuda import ext as huff
from boilerplate.kernels.ternary_cuda import ext as tern
assert huff() is not None, 'Huffman CUDA extension did not build (no GPU or no toolkit)'
assert tern() is not None, 'ternary CUDA extension did not build'
print('both CUDA extensions built')"
fi
if want tests; then
  # every test, with skip reasons; the CUDA tests must now run instead of skipping
  run tests python -m pytest -q -rs tests/
  run cuda_tests python -m pytest -q -rs tests/test_ternary_cuda.py tests/test_huffman_cuda.py \
      "tests/test_standard_engine.py::test_cuda_device_path_matches_cpu"
  grep -q "skipped" "$OUT/cuda_tests.log" && log "cuda_tests: some CUDA tests SKIPPED, the GPU path is not verified"
fi
if want kernels; then
  run kernels python scripts/eval/ternary_kernel_bench.py --device cuda --out "$OUT/kernel_bench.json"
fi
if want engine && [ -d "$MODEL" ]; then
  # Boilerplate decode on the GPU: source BF16 and the absmean TQ1 PTQ export of the same model
  [ -d "$MODEL-absmean-tq1" ] || run export_ptq boilerplate export "$MODEL" "$MODEL-absmean-tq1" --policy configs/policy/std_absmean_tq1.yaml
  for d in "$MODEL" "$MODEL-absmean-tq1"; do
    for b in 1 8; do
      run "bench_$(basename "$d")_b$b" boilerplate bench "$d" --device cuda --batch "$b" --prompt-len 512 --gen-len 64
    done
  done
fi
if want probe && [ -d "$MODEL" ]; then
  # 20-step QAT throughput probe (no approval needed); then the estimate from the measured tok/s
  PROBE=configs/qat/smollm2_shakedown.yaml
  if [ -f work/data/fineweb_edu_smollm2.bin ]; then
    run probe boilerplate qat-train "$PROBE" --max-steps 20
    python - "$PROBE" "$OUT" <<'PY' | tee -a "$OUT/summary.txt"
import json, subprocess, sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
rows = [json.loads(l) for l in open(f"{cfg['out_dir']}/log.rank0.jsonl") if '"tok_s"' in l]
r = rows[-1]
print(f"probe: {r['tok_s']:.0f} tok/s, peak VRAM {r.get('max_vram_bytes', 0) / 2**30:.2f} GiB, loss {r['loss']:.3f}")
est = subprocess.run(["boilerplate", "qat-estimate", sys.argv[1], "--gpu", "rtx4060-laptop", "--tok-s", str(r["tok_s"])],
                     capture_output=True, text=True).stdout
open(f"{sys.argv[2]}/estimate_measured.json", "w").write(est)
e = json.loads(est)
print(f"measured estimate for the full shakedown: {e['wall_clock_hours']:.1f} h wall clock, {e['tokens'] / 1e9:.2f} B tokens")
PY
    log "probe: delete $(python -c "import yaml;print(yaml.safe_load(open('$PROBE'))['out_dir'])") before the real run so it starts at step 0"
  else
    log "probe: skipped, run scripts/launch/smollm2_shakedown.sh prepare first (needs the token file)"
  fi
fi
log "done; send back the folder $OUT"

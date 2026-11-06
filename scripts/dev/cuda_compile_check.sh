#!/usr/bin/env bash
# Compile the CUDA kernels to SASS on a machine without a GPU or CUDA toolkit (catches syntax,
# template and resource errors; it does not run them). Uses clang's CUDA front end plus CUDA
# headers, libdevice and ptxas from NVIDIA's pip wheels in a scratch directory.
#   scripts/dev/cuda_compile_check.sh [sm_89]
set -euo pipefail
ARCH=${1:-sm_89}
W=${CUDA_CHECK_DIR:-${TMPDIR:-/tmp}/boilerplate-cuda-check}
R=$W/root
if [ ! -x "$W/wheels/nvidia/cuda_nvcc/bin/ptxas" ]; then
  pip install -q --target "$W/wheels" --no-deps nvidia-cuda-nvcc-cu12 nvidia-cuda-runtime-cu12 \
    nvidia-cublas-cu12 nvidia-cusparse-cu12 nvidia-cusolver-cu12 nvidia-cuda-cccl-cu12 nvidia-curand-cu12
fi
mkdir -p "$R/include" "$R/nvvm" "$R/bin" "$R/stub/c10/cuda/impl"
for d in "$W"/wheels/nvidia/*/include; do cp -r --update=none "$d"/. "$R/include/"; done
cp -r --update=none "$W/wheels/nvidia/cuda_nvcc/nvvm/libdevice" "$R/nvvm/"
printf '#!/bin/sh\necho "Cuda compilation tools, release 12.3, V12.3.107"\n' > "$R/bin/nvcc"; chmod +x "$R/bin/nvcc"
# generated in CUDA builds of torch only; the CPU wheel lacks it
printf '#pragma once\n#define C10_CUDA_BUILD_SHARED_LIBS\n' > "$R/stub/c10/cuda/impl/cuda_cmake_macros.h"
# clang 18 does not know CUDA 12.9's launch API: declare the legacy entry point for its device pass
printf '#include <cuda_runtime.h>\nextern "C" cudaError_t cudaConfigureCall(dim3, dim3, size_t = 0, cudaStream_t = 0);\n' > "$R/stub/shim.h"
TI=$(python -c "import torch,os;print(os.path.dirname(torch.__file__)+'/include')")
PI=$(python -c "import sysconfig;print(sysconfig.get_paths()['include'])")
cd "$(dirname "$0")/../.."
for f in boilerplate/kernels/csrc/*_cuda.cpp; do
  g++ -std=c++20 -fsyntax-only -DTORCH_EXTENSION_NAME=check -I"$TI" -I"$TI/torch/csrc/api/include" -I"$PI" "$f" && echo "== $f: host binding OK"
done
for f in boilerplate/kernels/csrc/*.cu; do
  echo "== $f ($ARCH)"
  clang++ -x cuda --cuda-path="$R" --cuda-gpu-arch="$ARCH" -Wno-unknown-cuda-version -std=c++20 -O2 \
    -DTORCH_EXTENSION_NAME=check -I"$R/stub" -I"$TI" -I"$TI/torch/csrc/api/include" -I"$PI" \
    --cuda-host-only -fsyntax-only "$f"
  clang++ -x cuda --cuda-path="$R" --cuda-gpu-arch="$ARCH" -Wno-unknown-cuda-version -std=c++20 -O2 \
    -DTORCH_EXTENSION_NAME=check -I"$R/stub" -I"$TI" -I"$TI/torch/csrc/api/include" -I"$PI" \
    -include "$R/stub/shim.h" --cuda-device-only -S -o "$W/$(basename "$f" .cu).ptx" "$f"
  "$W/wheels/nvidia/cuda_nvcc/bin/ptxas" -arch="$ARCH" -v "$W/$(basename "$f" .cu).ptx" -o /dev/null 2>&1 \
    | grep -E "Compiling|registers|spill"
done
echo "all CUDA sources compile for $ARCH"

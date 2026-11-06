// Ternary GEMM / grouped (MoE) GEMM on CUDA, int8 activations, TQ2 or TQ1 packed weights.
//
// Same contract as the C++ CPU kernel (boilerplate_cpu.cpp) and the PyTorch reference
// (boilerplate/kernels/ternary_cuda.py::ternary_gemm_ref):
//   out[t, n] += x_scale[t] * sum_g w_scale[e, n, g] * acc(t, n, g)
//   acc(t, n, g) = sum_{k in group g} x[t, k] * trit[e, n, k]          (exact int32)
// where e is the expert that owns token row t: rows are sorted by expert and expert e owns rows
// offsets[e] .. offsets[e+1]. The dense GEMM is the grouped GEMM with one expert.
//
// Layout: one warp per output row n, TOK_TILE token rows per block in registers. Lanes stride
// over k inside a group, decode the trit from the packed byte (TQ2: shift/mask; TQ1: a 243 x 5
// table staged in shared memory), add or subtract the int8 activation (no multiply), then the
// warp reduces each token's int32 group sum and lane 0 applies the group scale. Decode-time GEMV
// (one to a few tokens) is bandwidth bound on the packed weights, which every warp reads once
// per token tile.
//
// Status: compiled to sm_89 SASS without a GPU (clang 18 CUDA front end + ptxas 12.9, see
// scripts/dev/cuda_compile_check.sh); never run. tests/test_ternary_cuda.py checks it against the
// reference on any CUDA machine.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cstdint>

namespace {

constexpr int WARPS = 8;     // output rows per block
constexpr int TOK_TILE = 8;  // token rows per block

template <int FMT>
__device__ __forceinline__ int trit(const uint8_t* row, int64_t k, const int8_t* lut1) {
  if (FMT == 2) {
    const uint8_t b = row[k >> 2];
    return static_cast<int>((b >> ((k & 3) * 2)) & 3) - 1;
  } else {
    const uint8_t b = row[k / 5];
    return lut1[b * 5 + static_cast<int>(k % 5)];
  }
}

template <int FMT>
__global__ void ternary_grouped_kernel(float* __restrict__ out, const int8_t* __restrict__ x,
                                       const float* __restrict__ x_scale, const int64_t* __restrict__ offsets,
                                       const uint8_t* __restrict__ packed, const float* __restrict__ w_scale,
                                       const int8_t* __restrict__ lut1_global, int64_t N, int64_t K, int64_t Kp,
                                       int64_t G, int64_t group) {
  __shared__ int8_t lut1[256 * 5];
  if (FMT == 1) {
    for (int i = threadIdx.x; i < 256 * 5; i += blockDim.x) lut1[i] = lut1_global[i];
    __syncthreads();
  }
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int64_t e = blockIdx.y;
  const int64_t n = static_cast<int64_t>(blockIdx.x) * WARPS + warp;
  const int64_t row0 = offsets[e] + static_cast<int64_t>(blockIdx.z) * TOK_TILE;
  const int64_t row_end = offsets[e + 1];
  if (n >= N || row0 >= row_end) return;  // no __syncthreads below, so early exit is safe
  const int rows = static_cast<int>(min(static_cast<int64_t>(TOK_TILE), row_end - row0));

  const uint8_t* wrow = packed + (e * N + n) * Kp;
  const float* srow = w_scale + (e * N + n) * G;
  float y[TOK_TILE];
#pragma unroll
  for (int r = 0; r < TOK_TILE; ++r) y[r] = 0.f;

  for (int64_t g = 0; g < G; ++g) {
    const int64_t k0 = g * group, k1 = min(K, k0 + group);
    int acc[TOK_TILE];
#pragma unroll
    for (int r = 0; r < TOK_TILE; ++r) acc[r] = 0;
    for (int64_t k = k0 + lane; k < k1; k += 32) {
      const int t = trit<FMT>(wrow, k, lut1);
      if (t == 0) continue;
#pragma unroll
      for (int r = 0; r < TOK_TILE; ++r) {
        if (r < rows) {
          const int v = x[(row0 + r) * K + k];
          acc[r] += t > 0 ? v : -v;
        }
      }
    }
#pragma unroll
    for (int r = 0; r < TOK_TILE; ++r) {
      int a = acc[r];
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) a += __shfl_down_sync(0xffffffffu, a, off);
      if (lane == 0) y[r] += srow[g] * static_cast<float>(a);
    }
  }
  if (lane == 0) {
#pragma unroll
    for (int r = 0; r < TOK_TILE; ++r) {  // unrolled with a guard so y[] stays in registers
      if (r < rows) out[(row0 + r) * N + n] += y[r] * x_scale[row0 + r];
    }
  }
}

}  // namespace

// out [T, N] f32 (accumulated into); x [T, K] int8 sorted by expert; x_scale [T] f32;
// offsets [E + 1] int64 (device); packed [E, N, Kp] uint8; w_scale [E, N, G] f32; lut1 [256, 5] int8.
// max_rows: the largest offsets[e+1] - offsets[e] (sizes the launch grid; computed by the caller).
void ternary_grouped_gemm_cuda(torch::Tensor out, torch::Tensor x, torch::Tensor x_scale, torch::Tensor offsets,
                               torch::Tensor packed, torch::Tensor w_scale, int64_t K, int64_t group, int64_t fmt,
                               torch::Tensor lut1, int64_t max_rows) {
  for (const auto* t : {&out, &x, &x_scale, &offsets, &packed, &w_scale, &lut1}) {
    TORCH_CHECK(t->is_cuda() && t->is_contiguous(), "ternary GEMM tensors must be contiguous CUDA tensors");
    TORCH_CHECK(t->device() == out.device(), "ternary GEMM tensors must share one device");
  }
  TORCH_CHECK(out.scalar_type() == torch::kFloat32 && x.scalar_type() == torch::kInt8 &&
              x_scale.scalar_type() == torch::kFloat32 && offsets.scalar_type() == torch::kInt64 &&
              packed.scalar_type() == torch::kUInt8 && w_scale.scalar_type() == torch::kFloat32 &&
              lut1.scalar_type() == torch::kInt8, "unexpected ternary GEMM dtype");
  TORCH_CHECK(packed.dim() == 3 && w_scale.dim() == 3 && x.dim() == 2 && out.dim() == 2, "bad ranks");
  TORCH_CHECK(fmt == 1 || fmt == 2, "fmt must be 1 (TQ1) or 2 (TQ2)");
  const int64_t E = packed.size(0), N = packed.size(1), Kp = packed.size(2), G = w_scale.size(2);
  TORCH_CHECK(x.size(1) == K && Kp * (fmt == 2 ? 4 : 5) >= K && G * group >= K && offsets.numel() == E + 1,
              "inconsistent ternary GEMM shapes");
  TORCH_CHECK(out.size(0) == x.size(0) && out.size(1) == N && x_scale.numel() == x.size(0), "bad output shape");
  if (max_rows <= 0 || N == 0) return;
  c10::cuda::CUDAGuard guard(out.device());
  const dim3 grid((N + WARPS - 1) / WARPS, E, (max_rows + TOK_TILE - 1) / TOK_TILE);
  auto stream = c10::cuda::getCurrentCUDAStream();
  auto launch = [&](auto kernel) {
    kernel<<<grid, WARPS * 32, 0, stream>>>(out.data_ptr<float>(), x.data_ptr<int8_t>(), x_scale.data_ptr<float>(),
                                            offsets.data_ptr<int64_t>(), packed.data_ptr<uint8_t>(),
                                            w_scale.data_ptr<float>(), lut1.data_ptr<int8_t>(), N, K, Kp, G, group);
  };
  if (fmt == 2) launch(ternary_grouped_kernel<2>); else launch(ternary_grouped_kernel<1>);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

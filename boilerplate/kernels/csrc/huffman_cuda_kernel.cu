#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cstdint>
#include "huffman_validation.h"

// Canonical MSB-first codes. Each CUDA block owns one independently indexed symbol block.
// Threads cooperatively stage the 4096-entry table; one thread walks the variable-length codes
// within this block in order. For 2-byte inputs, interleave the raw low byte with decoded high.
__global__ void huff_decode_kernel(const uint8_t* stream, const int64_t* offsets, int64_t n,
                                   int64_t symbols_per_block, const int32_t* lut, int maxlen,
                                   const uint8_t* raw, uint8_t* out, int64_t stream_bytes, int* error) {
  __shared__ int32_t table[4096];
  const int table_size = 1 << maxlen;
  for (int i = threadIdx.x; i < table_size; i += blockDim.x) table[i] = lut[i];
  __syncthreads();
  if (threadIdx.x != 0) return;

  const int64_t first = static_cast<int64_t>(blockIdx.x) * symbols_per_block;
  const int64_t end = first + ((n - first < symbols_per_block) ? n - first : symbols_per_block);
  int64_t pos = offsets[blockIdx.x];
  for (int64_t i = first; i < end; ++i) {
    const int64_t byte = pos >> 3;
    if (pos >= offsets[blockIdx.x + 1] || byte > stream_bytes - 4) {
      atomicExch(error, 1); return;
    }
    const uint32_t word = (static_cast<uint32_t>(stream[byte]) << 24) |
                          (static_cast<uint32_t>(stream[byte + 1]) << 16) |
                          (static_cast<uint32_t>(stream[byte + 2]) << 8) |
                          static_cast<uint32_t>(stream[byte + 3]);
    const uint32_t peek = (word << (pos & 7)) >> (32 - maxlen);
    const int32_t entry = table[peek];
    const int length = entry & 0xff;
    if (!length || length > offsets[blockIdx.x + 1] - pos) {
      atomicExch(error, 1); return;
    }
    const uint8_t symbol = static_cast<uint8_t>(entry >> 8);
    pos += length;
    if (raw) {
      out[2 * i] = raw[i];
      out[2 * i + 1] = symbol;
    } else {
      out[i] = symbol;
    }
  }
  if (pos != offsets[blockIdx.x + 1]) atomicExch(error, 1);
}

void huff_decode_into_cuda(torch::Tensor stream, torch::Tensor offsets, int64_t n,
                           int64_t block, torch::Tensor lut, int64_t maxlen,
                           torch::Tensor raw, torch::Tensor out) {
  TORCH_CHECK(stream.is_cuda() && offsets.is_cuda() && lut.is_cuda() && raw.is_cuda() && out.is_cuda(),
              "Huffman inputs and output must be on CUDA");
  TORCH_CHECK(stream.device() == offsets.device() && stream.device() == lut.device() &&
              stream.device() == raw.device() && stream.device() == out.device(),
              "Huffman inputs and output must be on the same CUDA device");
  TORCH_CHECK(stream.is_contiguous() && offsets.is_contiguous() && lut.is_contiguous() &&
              raw.is_contiguous() && out.is_contiguous(), "Huffman tensors must be contiguous");
  TORCH_CHECK(stream.scalar_type() == torch::kUInt8 && offsets.scalar_type() == torch::kInt64 &&
              lut.scalar_type() == torch::kInt32 && raw.scalar_type() == torch::kUInt8 &&
              out.scalar_type() == torch::kUInt8, "unexpected Huffman tensor dtype");
  TORCH_CHECK(n >= 0 && block > 0 && maxlen > 0 && maxlen <= 12, "invalid Huffman dimensions");
  const int64_t blocks = check_huffman_layout(stream.numel(), offsets.cpu(), n, block, lut.cpu(), maxlen);
  TORCH_CHECK(blocks <= 2147483647, "too many CUDA Huffman blocks");
  TORCH_CHECK(raw.numel() == 0 || raw.numel() >= n, "raw low-byte tensor is too small");
  TORCH_CHECK(out.numel() >= n * (raw.numel() ? 2 : 1), "Huffman output buffer is too small");
  if (!n) return;
  TORCH_CHECK(stream.numel() >= 4, "Huffman stream lacks padding");
  c10::cuda::CUDAGuard guard(stream.device());
  auto error = torch::zeros({1}, out.options().dtype(torch::kInt32));
  huff_decode_kernel<<<blocks, 128, 0, c10::cuda::getCurrentCUDAStream()>>>(
      stream.data_ptr<uint8_t>(), offsets.data_ptr<int64_t>(), n, block, lut.data_ptr<int32_t>(),
      static_cast<int>(maxlen), raw.numel() ? raw.data_ptr<uint8_t>() : nullptr, out.data_ptr<uint8_t>(),
      stream.numel(), error.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  TORCH_CHECK(error.item<int>() == 0, "invalid or truncated Huffman bitstream");
}

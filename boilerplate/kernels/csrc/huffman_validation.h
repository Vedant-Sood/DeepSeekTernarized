#pragma once

#include <torch/extension.h>
#include <cstdint>
#include <algorithm>
#include <limits>

inline void check_cpu_tensor(const torch::Tensor& t, torch::ScalarType dtype) {
  TORCH_CHECK(t.device().is_cpu() && t.is_contiguous() && t.scalar_type() == dtype,
              "expected a contiguous CPU tensor with the required dtype");
}

// Validate metadata before allocating output or entering a parallel decoder. Offsets can
// start after zero when the engine selects blocks from a larger stream.
inline int64_t check_huffman_layout(int64_t stream_bytes, const torch::Tensor& offsets,
                                    int64_t n, int64_t block, const torch::Tensor& lut,
                                    int64_t maxlen) {
  check_cpu_tensor(offsets, torch::kInt64);
  check_cpu_tensor(lut, torch::kInt32);
  TORCH_CHECK(n >= 0 && n <= std::numeric_limits<int64_t>::max() / 12 &&
              block > 0 && maxlen > 0 && maxlen <= 12, "invalid Huffman dimensions");
  TORCH_CHECK(stream_bytes >= 4 && stream_bytes <= std::numeric_limits<int64_t>::max() / 8,
              "Huffman stream lacks padding or is too large");
  const int64_t bit_limit = (stream_bytes - 3) * 8;
  TORCH_CHECK(n <= bit_limit, "Huffman output exceeds the available encoded bits");
  const int64_t blocks = n / block + (n % block != 0);
  TORCH_CHECK(offsets.dim() == 1 && offsets.numel() == blocks + 1 &&
              lut.dim() == 1 && lut.numel() == (1LL << maxlen), "incomplete Huffman metadata");
  const auto* off = offsets.data_ptr<int64_t>();
  for (int64_t b = 0; b <= blocks; ++b)
    TORCH_CHECK(off[b] >= 0 && off[b] <= bit_limit, "Huffman offset outside stream");
  for (int64_t b = 0; b < blocks; ++b) {
    const int64_t count = std::min(block, n - b * block);
    TORCH_CHECK(off[b + 1] >= off[b] && off[b + 1] - off[b] >= count &&
                off[b + 1] - off[b] <= count * maxlen, "invalid Huffman block offsets");
  }
  const auto* table = lut.data_ptr<int32_t>();
  for (int64_t i = 0; i < lut.numel(); ++i)
    TORCH_CHECK(table[i] >= 0 && table[i] <= 0xffff && (table[i] & 255) <= maxlen,
                "invalid Huffman lookup entry");
  return blocks;
}

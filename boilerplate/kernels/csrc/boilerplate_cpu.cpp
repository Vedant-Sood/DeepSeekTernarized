// Boilerplate CPU kernels: ternary GEMM (TQ2 / TQ1, multiply-free inner loop) and Huffman
// encode/decode for Mode B. Built on first use by boilerplate/kernels/cpu.py with -O3 -march=native.
//
// Ternary GEMM, int8 activations:  out[m, n] += x_scale[m] * sum_g w_scale[n, g] * acc(m, n, g)
//   acc(m, n, g) = sum_{k in group g} (t[n,k] == +1 ? x[m,k] : 0) - (t[n,k] == -1 ? x[m,k] : 0)
// The inner loop only adds and subtracts (AVX2: _mm256_sign_epi8 + widening adds; elsewhere a
// select-and-add the compiler vectorizes). One multiply per group, one per output.
#include <torch/extension.h>
#include <cstdint>
#include <vector>
#include <cstring>
#ifdef __AVX2__
#include <immintrin.h>
#endif

static inline void decode_row(const uint8_t* p, int64_t K, int fmt, int8_t* out, const int8_t* lut1) {
  if (fmt == 2) {
    int64_t nb = (K + 3) / 4;
    for (int64_t b = 0; b < nb; ++b) {
      uint8_t v = p[b];
      int8_t* o = out + 4 * b;
      o[0] = (int8_t)((v & 3) - 1); o[1] = (int8_t)(((v >> 2) & 3) - 1);
      o[2] = (int8_t)(((v >> 4) & 3) - 1); o[3] = (int8_t)(((v >> 6) & 3) - 1);
    }
  } else {
    int64_t nb = (K + 4) / 5;
    for (int64_t b = 0; b < nb; ++b) std::memcpy(out + 5 * b, lut1 + 5 * p[b], 5);
  }
}

static inline int32_t tdot_i8(const int8_t* x, const int8_t* t, int64_t n) {
  int64_t i = 0;
  int32_t acc = 0;
#ifdef __AVX2__
  __m256i a16 = _mm256_setzero_si256();
  int blocks = 0;
  for (; i + 32 <= n; i += 32) {
    __m256i xv = _mm256_loadu_si256((const __m256i*)(x + i));
    __m256i tv = _mm256_loadu_si256((const __m256i*)(t + i));
    __m256i s = _mm256_sign_epi8(xv, tv);  // +x, -x or 0: no multiply
    a16 = _mm256_add_epi16(a16, _mm256_cvtepi8_epi16(_mm256_castsi256_si128(s)));
    a16 = _mm256_add_epi16(a16, _mm256_cvtepi8_epi16(_mm256_extracti128_si256(s, 1)));
    if (++blocks == 64) {  // keep int16 lanes far from overflow (64 * 2 * 127 < 32767)
      __m256i lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(a16));
      __m256i hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(a16, 1));
      __m256i s32 = _mm256_add_epi32(lo, hi);
      alignas(32) int32_t buf[8];
      _mm256_store_si256((__m256i*)buf, s32);
      for (int j = 0; j < 8; ++j) acc += buf[j];
      a16 = _mm256_setzero_si256();
      blocks = 0;
    }
  }
  {
    __m256i lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(a16));
    __m256i hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(a16, 1));
    __m256i s32 = _mm256_add_epi32(lo, hi);
    alignas(32) int32_t buf[8];
    _mm256_store_si256((__m256i*)buf, s32);
    for (int j = 0; j < 8; ++j) acc += buf[j];
  }
#endif
  for (; i < n; ++i) acc += (t[i] > 0 ? x[i] : 0) - (t[i] < 0 ? x[i] : 0);
  return acc;
}

static inline float tdot_f32(const float* x, const int8_t* t, int64_t n) {
  float pos = 0.f, neg = 0.f;
  for (int64_t i = 0; i < n; ++i) {
    pos += t[i] > 0 ? x[i] : 0.f;
    neg += t[i] < 0 ? x[i] : 0.f;
  }
  return pos - neg;
}

// out [M, N] float32 (accumulated into). x: int8 [M, K] + x_scale [M], or float [M, K] (x_scale ignored).
void ternary_gemm(torch::Tensor out, torch::Tensor x, torch::Tensor x_scale, torch::Tensor packed,
                  torch::Tensor w_scale, int64_t K, int64_t group, int64_t fmt, torch::Tensor lut1) {
  TORCH_CHECK(out.is_contiguous() && x.is_contiguous() && packed.is_contiguous() && w_scale.is_contiguous());
  const int64_t M = x.size(0), N = packed.size(0), Kp = packed.size(1), G = w_scale.size(1);
  const bool i8 = x.scalar_type() == torch::kInt8;
  const uint8_t* P = packed.data_ptr<uint8_t>();
  const float* S = w_scale.data_ptr<float>();
  float* O = out.data_ptr<float>();
  const int8_t* L = lut1.data_ptr<int8_t>();
  const int8_t* X8 = i8 ? x.data_ptr<int8_t>() : nullptr;
  const float* XF = i8 ? nullptr : x.data_ptr<float>();
  const float* XS = i8 ? x_scale.data_ptr<float>() : nullptr;
  const int64_t Kpad = Kp * (fmt == 2 ? 4 : 5);
#pragma omp parallel
  {
    std::vector<int8_t> trow(Kpad + 64);
#pragma omp for schedule(static)
    for (int64_t n = 0; n < N; ++n) {
      decode_row(P + n * Kp, K, (int)fmt, trow.data(), L);
      const float* s = S + n * G;
      for (int64_t m = 0; m < M; ++m) {
        float y = 0.f;
        for (int64_t g = 0; g < G; ++g) {
          int64_t k0 = g * group, len = std::min(group, K - k0);
          float a = i8 ? (float)tdot_i8(X8 + m * K + k0, trow.data() + k0, len)
                       : tdot_f32(XF + m * K + k0, trow.data() + k0, len);
          y += s[g] * a;
        }
        O[m * N + n] += i8 ? y * XS[m] : y;
      }
    }
  }
}

// ---------------------------------------------------------------- Huffman (canonical, MSB-first)
// codes[s] holds the code in its low lens[s] bits. Stream is split into blocks of `block` symbols;
// offsets[b] = bit position where block b starts, so blocks decode independently (GPU-style).
std::vector<torch::Tensor> huff_encode(torch::Tensor sym, torch::Tensor codes, torch::Tensor lens, int64_t block) {
  const int64_t n = sym.numel();
  const uint8_t* s = sym.data_ptr<uint8_t>();
  const int32_t* C = codes.data_ptr<int32_t>();
  const uint8_t* Ln = lens.data_ptr<uint8_t>();
  const int64_t nb = (n + block - 1) / block;
  auto offsets = torch::empty({nb + 1}, torch::kInt64);
  int64_t* off = offsets.data_ptr<int64_t>();
  int64_t bits = 0;
  for (int64_t i = 0; i < n; ++i) {
    if (i % block == 0) off[i / block] = bits;
    bits += Ln[s[i]];
  }
  off[nb] = bits;
  auto out = torch::zeros({(bits + 7) / 8 + 8}, torch::kUInt8);
  uint8_t* o = out.data_ptr<uint8_t>();
  int64_t pos = 0;
  for (int64_t i = 0; i < n; ++i) {
    uint32_t c = (uint32_t)C[s[i]];
    int l = Ln[s[i]];
    for (int b = l - 1; b >= 0; --b, ++pos)
      if ((c >> b) & 1) o[pos >> 3] |= (uint8_t)(0x80 >> (pos & 7));
  }
  return {out, offsets};
}

// lut: [1 << maxlen] entries, (symbol << 8) | length, indexed by the next maxlen bits.
torch::Tensor huff_decode(torch::Tensor stream, torch::Tensor offsets, int64_t n, int64_t block, torch::Tensor lut, int64_t maxlen) {
  auto out = torch::empty({n}, torch::kUInt8);
  uint8_t* o = out.data_ptr<uint8_t>();
  const uint8_t* in = stream.data_ptr<uint8_t>();
  const int64_t* off = offsets.data_ptr<int64_t>();
  const int32_t* T = lut.data_ptr<int32_t>();
  const int64_t nb = (n + block - 1) / block;
  const uint32_t mask = (1u << maxlen) - 1;
#pragma omp parallel for schedule(dynamic, 16)
  for (int64_t b = 0; b < nb; ++b) {
    int64_t pos = off[b];
    int64_t end = std::min(n, (b + 1) * block);
    for (int64_t i = b * block; i < end; ++i) {
      int64_t byte = pos >> 3;
      uint32_t w = ((uint32_t)in[byte] << 24) | ((uint32_t)in[byte + 1] << 16) | ((uint32_t)in[byte + 2] << 8) | (uint32_t)in[byte + 3];
      uint32_t peek = (w << (pos & 7)) >> (32 - maxlen);
      int32_t e = T[peek & mask];
      o[i] = (uint8_t)(e >> 8);
      pos += e & 0xFF;
    }
  }
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ternary_gemm", &ternary_gemm);
  m.def("huff_encode", &huff_encode);
  m.def("huff_decode", &huff_decode);
}

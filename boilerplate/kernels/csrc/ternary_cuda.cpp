#include <torch/extension.h>

void ternary_grouped_gemm_cuda(torch::Tensor out, torch::Tensor x, torch::Tensor x_scale, torch::Tensor offsets,
                               torch::Tensor packed, torch::Tensor w_scale, int64_t K, int64_t group, int64_t fmt,
                               torch::Tensor lut1, int64_t max_rows);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ternary_grouped_gemm", &ternary_grouped_gemm_cuda,
        "Grouped (per-expert) ternary GEMM with int8 activations; the dense GEMM is one group");
}

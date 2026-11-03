#include <torch/extension.h>

void huff_decode_into_cuda(torch::Tensor stream, torch::Tensor offsets, int64_t n,
                           int64_t block, torch::Tensor lut, int64_t maxlen,
                           torch::Tensor raw, torch::Tensor out);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("huff_decode_into", &huff_decode_into_cuda,
        "Decode Huffman blocks into a caller-owned CUDA byte buffer");
}

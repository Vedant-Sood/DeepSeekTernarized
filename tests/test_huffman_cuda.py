"""Mode B CUDA parity and reusable output buffers."""

import pytest
import torch

from boilerplate.format import entropy
from boilerplate.kernels.cpu import ext as cpu_ext
from boilerplate.kernels.cuda import ext as cuda_ext


def _bytes_as(dtype, n):
    size = torch.empty((), dtype=dtype).element_size()
    raw = torch.randint(0, 256, (n * size,), dtype=torch.uint8, generator=torch.Generator().manual_seed(41))
    return raw.view(dtype)


@pytest.mark.skipif(cpu_ext() is None, reason="C++ extension unavailable")
def test_cpu_decode_reuses_output_without_touching_suffix():
    tensor = _bytes_as(torch.bfloat16, 4101)
    parts = entropy.encode(tensor)
    out = torch.full((tensor.numel() * 2 + 17,), 0xa5, dtype=torch.uint8)
    decoded = entropy.decode(parts, tensor.numel(), out=out)
    assert decoded.data_ptr() == out.data_ptr()
    assert torch.equal(decoded, tensor.view(torch.uint8))
    assert torch.equal(out[-17:], torch.full((17,), 0xa5, dtype=torch.uint8))
    with pytest.raises(ValueError, match="out must be"):
        entropy.decode(parts, tensor.numel(), out=out[:-18])


def test_cuda_extension_absence_is_clean():
    if torch.version.cuda is None or not torch.cuda.is_available():
        assert cuda_ext() is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU unavailable")
@pytest.mark.parametrize("dtype", [torch.uint8, torch.int8, torch.float4_e2m1fn_x2,
                                   torch.float8_e4m3fn, torch.float8_e8m0fnu,
                                   torch.bfloat16, torch.float16])
def test_gpu_huffman_matches_cpu_bytes(dtype):
    if cuda_ext() is None:
        pytest.skip("CUDA toolkit unavailable or CUDA extension disabled")
    tensor = _bytes_as(dtype, 10007)  # two full blocks and a partial block
    parts = entropy.encode(tensor)
    expected = entropy.decode(parts, tensor.numel())
    gpu_parts = {name: part.cuda() for name, part in parts.items()}
    out = torch.full((expected.numel() + 17,), 0xa5, dtype=torch.uint8, device="cuda")
    decoded = entropy.decode(gpu_parts, tensor.numel(), out=out)
    assert decoded.data_ptr() == out.data_ptr()
    assert torch.equal(decoded.cpu(), expected)
    assert torch.equal(out[-17:].cpu(), torch.full((17,), 0xa5, dtype=torch.uint8))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU unavailable")
@pytest.mark.parametrize("n", [1, 4095, 4096, 4097])
def test_gpu_huffman_block_boundaries(n):
    if cuda_ext() is None:
        pytest.skip("CUDA toolkit unavailable or CUDA extension disabled")
    tensor = _bytes_as(torch.uint8, n)
    parts = entropy.encode(tensor)
    actual = entropy.decode({name: part.cuda() for name, part in parts.items()}, n).cpu()
    assert torch.equal(actual, tensor)

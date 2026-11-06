"""Native entry points must reject malformed tensors before dereferencing their storage."""

import pytest
import torch

from boilerplate.format import entropy
from boilerplate.kernels.cpu import ext
from boilerplate.ternary.pack import _TQ1_LUT


@pytest.fixture(scope="module")
def native():
    extension = ext()
    if extension is None:
        pytest.skip("C++ extension disabled")
    return extension


def _gemm_args():
    return dict(out=torch.zeros(1, 2), x=torch.ones(1, 5), x_scale=torch.ones(1),
                packed=torch.ones(2, 2, dtype=torch.uint8), w_scale=torch.ones(2, 2),
                K=5, group=3, fmt=2, lut1=_TQ1_LUT)


@pytest.mark.parametrize("key,value", [
    ("K", 0), ("K", 2**62), ("group", 0), ("fmt", 0),
    ("x", torch.ones(1, 4)), ("x", torch.ones(1, 5, dtype=torch.float64)),
    ("out", torch.zeros(1, 1)), ("out", torch.zeros(1, 2, dtype=torch.float64)),
    ("packed", torch.ones(2, 1, dtype=torch.uint8)), ("packed", torch.ones(2, 2)),
    ("w_scale", torch.ones(2, 1)), ("w_scale", torch.ones(1, 2)),
    ("x_scale", torch.ones(0)), ("lut1", torch.ones(1, dtype=torch.int8)),
    ("packed", torch.ones(2, 4, dtype=torch.uint8)[:, ::2]),
])
def test_ternary_gemm_rejects_unsafe_shapes(native, key, value):
    args = _gemm_args()
    args[key] = value
    with pytest.raises(RuntimeError):
        native.ternary_gemm(*args.values())


def _decode_args():
    return dict(stream=torch.zeros(8, dtype=torch.uint8), offsets=torch.tensor([0, 1]),
                n=1, block=1, lut=torch.tensor([1, 257], dtype=torch.int32), maxlen=1)


@pytest.mark.parametrize("key,value", [
    ("stream", torch.zeros(3, dtype=torch.uint8)), ("stream", torch.zeros(8)),
    ("offsets", torch.tensor([-1, 0])), ("offsets", torch.tensor([1000, 1001])),
    ("offsets", torch.tensor([1, 0])), ("offsets", torch.tensor([0])),
    ("offsets", torch.tensor([0., 1.])), ("n", -1), ("n", 2**62),
    ("block", 0), ("block", -1), ("maxlen", 0), ("maxlen", 13),
    ("lut", torch.zeros(3, dtype=torch.int32)),
    ("lut", torch.tensor([-1, 257], dtype=torch.int32)),
    ("lut", torch.tensor([2, 257], dtype=torch.int32)),
    ("lut", torch.zeros(2, dtype=torch.int32)),
])
def test_huffman_decode_rejects_unsafe_metadata(native, key, value):
    args = _decode_args()
    args[key] = value
    with pytest.raises(RuntimeError):
        native.huff_decode(*args.values())


@pytest.mark.parametrize("length,end", [(2, 1), (1, 2)])
def test_huffman_decode_detects_overrun_and_unused_block_bits(native, length, end):
    args = _decode_args()
    args.update(maxlen=2, lut=torch.full((4,), length, dtype=torch.int32),
                offsets=torch.tensor([0, end]))
    with pytest.raises(RuntimeError, match="bitstream"):
        native.huff_decode(*args.values())


@pytest.mark.parametrize("case", ["block", "codes", "lens", "long_code", "missing_code", "dtype", "strided"])
def test_huffman_encode_rejects_unsafe_inputs(native, case):
    sym = torch.zeros(4, dtype=torch.uint8)
    codes = torch.zeros(256, dtype=torch.int32)
    lens = torch.ones(256, dtype=torch.uint8)
    block = 4
    if case == "block":
        block = 0
    elif case == "codes":
        codes = codes[:255]
    elif case == "lens":
        lens = lens[:255]
    elif case == "long_code":
        lens[0] = 13
    elif case == "missing_code":
        lens[0] = 0
    elif case == "dtype":
        sym = sym.float()
    elif case == "strided":
        sym = torch.zeros(8, dtype=torch.uint8)[::2]
    with pytest.raises(RuntimeError):
        native.huff_encode(sym, codes, lens, block)


@pytest.mark.parametrize("maxlen", [-1, 0, 13, 2**62])
def test_python_rejects_unbounded_lookup_tables(maxlen):
    with pytest.raises(ValueError, match="maxlen"):
        entropy.decode_table(torch.ones(256, dtype=torch.uint8), maxlen)


def test_python_rejects_oversubscribed_codes_and_impossible_limits():
    lens = torch.zeros(256, dtype=torch.uint8)
    lens[:3] = 1
    with pytest.raises(ValueError, match="oversubscribed"):
        entropy.decode_table(lens, 12)
    with pytest.raises(ValueError, match="cannot represent"):
        entropy.code_lengths(torch.ones(256, dtype=torch.int64), maxlen=1)


def test_python_rejects_unbounded_decode_allocation():
    parts = {"stream": torch.zeros(8, dtype=torch.uint8), "offsets": torch.tensor([0, 1]),
             "lens": torch.ones(256, dtype=torch.uint8)}
    with pytest.raises(ValueError, match="available encoded bits"):
        entropy.decode(parts, 2**62)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU unavailable")
@pytest.mark.parametrize("case", ["negative_offset", "past_end", "invalid_code", "short_bits"])
def test_cuda_rejects_invalid_offsets_and_bitstreams(case):
    from boilerplate.kernels.cuda import ext as cuda_ext
    extension = cuda_ext()
    if extension is None:
        pytest.skip("CUDA toolkit unavailable")
    args = _decode_args()
    if case == "negative_offset":
        args["offsets"] = torch.tensor([-1, 0])
    elif case == "past_end":
        args["offsets"] = torch.tensor([1000, 1001])
    elif case == "invalid_code":
        args["lut"] = torch.zeros(2, dtype=torch.int32)
    else:
        args.update(maxlen=2, lut=torch.full((4,), 2, dtype=torch.int32))
    tensors = {key: args[key].cuda() for key in ("stream", "offsets", "lut")}
    with pytest.raises(RuntimeError):
        extension.huff_decode_into(tensors["stream"], tensors["offsets"], args["n"], args["block"],
                                   tensors["lut"], args["maxlen"], torch.empty(0, device="cuda", dtype=torch.uint8),
                                   torch.empty(1, device="cuda", dtype=torch.uint8))

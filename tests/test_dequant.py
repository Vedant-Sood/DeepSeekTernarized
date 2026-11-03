import pytest
import torch

from boilerplate.checkpoint.dequant import (
    FP4_VALUES,
    dequant_fp4,
    dequant_fp8_block,
    infer_block,
    is_packed_fp4,
    pack_fp4,
    pow2_ceil,
    quant_fp4_group,
    quant_fp8_block,
    round_to_fp4,
    unpack_fp4,
)


def test_pow2_ceil_matches_definition():
    x = torch.tensor([0.3, 1.0, 1.0001, 448.0, 2.0**-20, 3.0])
    expected = torch.tensor([0.5, 1.0, 2.0, 512.0, 2.0**-20, 4.0])
    assert torch.equal(pow2_ceil(x), expected)


def test_fp4_round_ties_to_even():
    x = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, -0.75, 7.0, 0.1, -0.1])
    vals = FP4_VALUES[round_to_fp4(x)]
    assert vals.tolist() == [0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, -1.0, 6.0, 0.0, -0.0]


def test_fp4_round_is_nearest():
    x = torch.linspace(-6, 6, 2001)
    vals = FP4_VALUES[round_to_fp4(x)]
    grid = FP4_VALUES[:8]
    best = (x.abs().unsqueeze(-1) - grid).abs().min(-1).values
    assert torch.allclose((vals.abs() - x.abs()).abs(), best)


def test_pack_unpack_roundtrip_all_codes():
    codes = torch.arange(16).repeat(4, 2)  # [4, 32]
    packed = pack_fp4(codes)
    assert packed.dtype == torch.int8 and packed.shape == (4, 16)
    assert torch.equal(unpack_fp4(packed), FP4_VALUES[codes])
    # same layout as the reference convert.py: low nibble is the even element
    assert (packed.view(torch.uint8)[0, 0] & 0x0F).item() == 0 and (packed.view(torch.uint8)[0, 0] >> 4).item() == 1


def test_fp8_block_quant_dequant():
    torch.manual_seed(0)
    w = torch.randn(70, 96)  # non-multiple rows exercise the ceil
    q, s = quant_fp8_block(w, (32, 32))
    assert q.dtype == torch.float8_e4m3fn and s.dtype == torch.float8_e8m0fnu and s.shape == (3, 3)
    assert infer_block(tuple(q.shape), tuple(s.shape)) == (32, 32)
    w2 = dequant_fp8_block(q, s, torch.float32)
    rel = (w - w2).norm() / w.norm()
    assert rel < 0.05


def test_fp4_group_quant_dequant():
    torch.manual_seed(0)
    w = torch.randn(16, 128)
    q, s = quant_fp4_group(w, 32)
    assert q.shape == (16, 64) and s.shape == (16, 4)
    assert is_packed_fp4(tuple(q.shape), tuple(s.shape))
    w2 = dequant_fp4(q, s, torch.float32)
    assert (w - w2).norm() / w.norm() < 0.2
    # every dequantized value is a grid point times its group scale
    ratio = (w2.view(16, 4, 32) / s.float().unsqueeze(-1)).flatten()
    assert torch.isin(ratio, FP4_VALUES).all()


def test_infer_block_rejects_inconsistent():
    with pytest.raises(ValueError):
        infer_block((64, 64), (65, 2))

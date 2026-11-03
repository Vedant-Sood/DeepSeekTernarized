"""M2: ternary packing, fitting, kernels and Mode C export/engine on the tiny checkpoint."""

import itertools

import pytest
import torch

from boilerplate.kernels.cpu import ext
from boilerplate.ternary.fit import dequant_terms, fit_gptq, fit_residual, fit_term, roundtrip_check
from boilerplate.ternary.ops import PackedTerm, TernaryLinear, quant_act_int8
from boilerplate.ternary.pack import bits_per_weight, pack, packed_cols, unpack


@pytest.mark.parametrize("fmt", ["tq2", "tq1"])
@pytest.mark.parametrize("K", [1, 4, 5, 7, 128, 131, 640])
def test_pack_roundtrip(fmt, K):
    t = torch.randint(-1, 2, (9, K), dtype=torch.int8)
    p = pack(t, fmt)
    assert p.dtype == torch.uint8 and p.shape == (9, packed_cols(fmt, K))
    assert torch.equal(unpack(p, fmt, K), t)


def test_tq1_every_byte_pattern():
    # all 243 5-trit patterns map to distinct bytes and back
    pats = torch.tensor(list(itertools.product([-1, 0, 1], repeat=5)), dtype=torch.int8)
    p = pack(pats, "tq1")
    assert p.unique().numel() == 243 and int(p.max()) < 243
    assert torch.equal(unpack(p, "tq1", 5), pats)
    assert bits_per_weight("tq1") == 1.6 and bits_per_weight("tq2") == 2.0


def test_fit_term_is_optimal_per_group():
    # brute force over all thresholds equals the closed form
    torch.manual_seed(0)
    r = torch.randn(3, 16)
    t = fit_term(r, group=16, scale_dtype=torch.float32)
    err = (r - dequant_terms([t], 16, 16)).square().sum(-1)
    for i in range(3):
        best = min(
            (r[i] - torch.sign(r[i]) * (r[i].abs() >= th) * r[i].abs()[r[i].abs() >= th].mean()).square().sum()
            for th in r[i].abs()
        )
        assert err[i] <= best + 1e-5


def test_more_terms_lower_error():
    torch.manual_seed(0)
    w = torch.randn(64, 384) * torch.rand(64, 1)
    errs = [(w - dequant_terms(fit_residual(w, k, 128), 384, 128)).square().mean().item() for k in (1, 2, 3, 4)]
    assert all(a > b for a, b in zip(errs, errs[1:]))
    assert errs[-1] < 0.01 * w.square().mean()


def test_gptq_lowers_output_error():
    torch.manual_seed(0)
    N, K, M = 48, 256, 2048
    w = torch.randn(N, K)
    x = torch.randn(M, K) @ (torch.eye(K) + 0.3 * torch.randn(K, K) / K**0.5)  # correlated inputs
    x[:, :8] *= 6  # a few outlier channels
    H = x.T @ x
    plain = dequant_terms(fit_residual(w, 2, 128), K, 128)
    g = fit_gptq(w, H, 2, 128)
    assert all(roundtrip_check(t, "tq1") for t in g)
    gq = dequant_terms(g, K, 128)
    e_plain = (x @ (w - plain).T).square().mean()
    e_gptq = (x @ (w - gq).T).square().mean()
    assert e_gptq < 0.9 * e_plain


def test_int8_activation_quant():
    x = torch.randn(5, 300) * 3
    q, s = quant_act_int8(x)
    assert q.dtype == torch.int8 and q.abs().max() <= 127
    assert ((q.float() * s[:, None] - x).abs() <= s[:, None] / 2 + 1e-6).all()


@pytest.mark.skipif(ext() is None, reason="C++ extension unavailable")
@pytest.mark.parametrize("fmt", ["tq2", "tq1"])
@pytest.mark.parametrize("act", ["int8", "bf16"])
@pytest.mark.parametrize("K,group", [(256, 128), (640, 128), (200, 64), (1024, 256)])
def test_kernel_matches_dequantized(fmt, act, K, group):
    torch.manual_seed(K)
    w = torch.randn(37, K)
    terms = fit_residual(w, 2, group)
    lin = TernaryLinear([PackedTerm(pack(t.trits, fmt), t.scale) for t in terms], K, fmt, group, act=act, backend="cpp")
    ref = TernaryLinear(lin.terms, K, fmt, group, act=act, backend="torch")
    x = torch.randn(3, 5, K, dtype=torch.bfloat16)
    a, b = lin(x, out_dtype=torch.float32), ref(x, out_dtype=torch.float32)
    assert a.shape == (3, 5, 37)
    torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-3)

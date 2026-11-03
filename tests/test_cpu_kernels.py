import math

import torch

from boilerplate.checkpoint.dequant import dequant_fp4, dequant_fp8_block, quant_fp4_group, quant_fp8_block
from boilerplate.reference import cpu_kernels as K


def test_act_quant_pow2_scales_and_inplace_consistency():
    torch.manual_seed(0)
    x = torch.randn(3, 5, 128).bfloat16()
    q, s = K.act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu)
    assert q.dtype == torch.float8_e4m3fn and s.shape == (3, 5, 4) and s.dtype == torch.float8_e8m0fnu
    sf = s.float()
    assert torch.equal(torch.exp2(torch.log2(sf).round()), sf)  # powers of two
    deq = (q.float().unflatten(-1, (4, 32)) * sf.unsqueeze(-1)).flatten(-2).bfloat16()
    y = x.clone()
    K.act_quant(y, 32, "ue8m0", torch.float8_e8m0fnu, inplace=True)
    assert torch.equal(y, deq)
    assert ((y.float() - x.float()).norm() / x.float().norm()) < 0.05


def test_fp8_gemm_matches_dequantized_matmul():
    torch.manual_seed(0)
    w, ws = quant_fp8_block(torch.randn(96, 64) / 8, (32, 32))
    x = torch.randn(2, 7, 64)
    xq, xs = K.act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu)
    torch.set_default_dtype(torch.float32)
    out = K.fp8_gemm(xq, xs, w, ws, torch.float8_e8m0fnu, 32)
    xd = (xq.float().unflatten(-1, (2, 32)) * xs.float().unsqueeze(-1)).flatten(-2)
    ref = xd @ dequant_fp8_block(w, ws, torch.float32).t()
    assert out.shape == (2, 7, 96)
    assert torch.allclose(out, ref, rtol=1e-5, atol=1e-5)


def test_fp4_gemm_matches_dequantized_matmul():
    torch.manual_seed(0)
    w, ws = quant_fp4_group(torch.randn(48, 64), 32)
    x = torch.randn(5, 64)
    xq, xs = K.act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu)
    out = K.fp4_gemm(xq, xs, w.view(torch.float4_e2m1fn_x2), ws, torch.float8_e8m0fnu, 32)
    xd = (xq.float().unflatten(-1, (2, 32)) * xs.float().unsqueeze(-1)).flatten(-2)
    ref = xd @ dequant_fp4(w, ws, torch.float32).t()
    assert torch.allclose(out.float(), ref, rtol=1e-5, atol=1e-5)


def test_fp4_act_quant_roundtrip_and_scale_kinds():
    torch.manual_seed(0)
    x = torch.randn(4, 64)
    packed, s = K.fp4_act_quant(x, 32)
    assert packed.dtype == torch.float4_e2m1fn_x2 and packed.shape == (4, 32) and s.shape == (4, 2)
    y = x.clone()
    K.fp4_act_quant(y, 32, inplace=True)
    assert torch.allclose(y, dequant_fp4(packed.view(torch.int8), s, torch.float32))
    z = x.clone()
    K.fp4_act_quant(z, 16, inplace=True, scale_dtype=torch.float8_e4m3fn)
    assert ((z - x).norm() / x.norm()) < 0.2


def _naive_attn(q, kv, sink, idxs, scale):
    b, m, h, d = q.shape
    out = torch.zeros(b, m, h, d)
    for bi in range(b):
        for mi in range(m):
            sel = [i for i in idxs[bi, mi].tolist() if i >= 0]
            if not sel:
                continue
            k = kv[bi, sel].float()  # [k, d]
            logits = q[bi, mi].float() @ k.t() * scale  # [h, k]
            full = torch.cat([logits, sink.float().unsqueeze(-1)], -1).softmax(-1)
            out[bi, mi] = full[:, :-1] @ k
    return out


def test_sparse_attn_matches_naive_with_sink_and_empty_rows():
    torch.manual_seed(0)
    b, m, h, d, n = 2, 6, 4, 32, 10
    q = torch.randn(b, m, h, d).bfloat16()
    kv = torch.randn(b, n, d).bfloat16()
    sink = torch.randn(h)
    idxs = torch.randint(0, n, (b, m, 5)).int()
    idxs[0, 1, 2:] = -1
    idxs[1, 3] = -1  # no valid index at all -> zeros
    o = K.sparse_attn(q, kv, sink, idxs, 1 / math.sqrt(d))
    ref = _naive_attn(q, kv, sink, idxs, 1 / math.sqrt(d))
    assert o.dtype == torch.bfloat16
    assert torch.equal(o[1, 3].float(), torch.zeros(h, d))
    assert torch.allclose(o.float(), ref, atol=2e-2, rtol=2e-2)


def test_hc_sinkhorn_doubly_stochastic():
    torch.manual_seed(0)
    hc = 4
    mixes = torch.randn(2, 3, (2 + hc) * hc)
    pre, post, comb = K.hc_split_sinkhorn(mixes, torch.ones(3), torch.zeros((2 + hc) * hc), hc, 20, 1e-6)
    assert pre.shape == (2, 3, hc) and post.shape == (2, 3, hc) and comb.shape == (2, 3, hc, hc)
    assert torch.allclose(comb.sum(-2), torch.ones(2, 3, hc), atol=1e-5)
    assert torch.allclose(comb.sum(-1), torch.ones(2, 3, hc), atol=1e-3)
    assert ((post > 0) & (post < 2)).all() and (pre > 0).all()

"""Ternary GEMM backends agree: PyTorch reference, C++ CPU kernel, grouped fallback, CUDA (when present)."""

import pytest
import torch

from boilerplate.kernels.cpu import ext as cpu_ext
from boilerplate.kernels.ternary_cuda import (ext as cuda_ext, grouped_ternary_gemm, sort_by_expert,
                                              ternary_gemm, ternary_gemm_ref)
from boilerplate.ternary.ops import quant_act_int8
from boilerplate.ternary.pack import pack, unpack


def _weights(E, N, K, group, fmt, seed=0):
    g = torch.Generator().manual_seed(seed)
    trits = torch.randint(-1, 2, (E, N, K), generator=g).to(torch.int8)
    packed = torch.stack([pack(trits[e], fmt) for e in range(E)])
    scales = torch.rand(E, N, -(-K // group), generator=g).half().float() + 0.01
    return trits, packed, scales


def _dense(x, xs, trits, scales, group):
    K = x.size(1)
    w = trits.float() * scales.repeat_interleave(group, -1)[:, :K]
    return (x.float() * xs[:, None]) @ w.t()


@pytest.mark.parametrize("fmt", ["tq1", "tq2"])
@pytest.mark.parametrize("K,group", [(256, 128), (300, 64), (130, 128)])
def test_reference_matches_dense_and_cpu_kernel(fmt, K, group):
    trits, packed, scales = _weights(1, 24, K, group, fmt)
    assert torch.equal(unpack(packed[0], fmt, K), trits[0])
    x, xs = quant_act_int8(torch.randn(5, K, generator=torch.Generator().manual_seed(1)))
    ref = ternary_gemm_ref(x, xs, packed[0], scales[0], K, group, fmt)
    assert torch.allclose(ref, _dense(x, xs, trits[0], scales[0], group), rtol=1e-5, atol=1e-4)
    if cpu_ext() is not None:
        assert torch.allclose(ternary_gemm(x, xs, packed[0], scales[0], K, group, fmt, backend="cpp"), ref,
                              rtol=1e-5, atol=1e-4)
    assert torch.equal(ternary_gemm(x, xs, packed[0], scales[0], K, group, fmt, backend="torch"), ref)


@pytest.mark.parametrize("fmt", ["tq1", "tq2"])
def test_grouped_matches_per_expert_and_routing_sort(fmt):
    E, N, K, group, T, k = 6, 16, 192, 64, 9, 2
    trits, packed, scales = _weights(E, N, K, group, fmt, seed=3)
    g = torch.Generator().manual_seed(4)
    x, xs = quant_act_int8(torch.randn(T, K, generator=g))
    ids = torch.stack([torch.randperm(E, generator=g)[:k] for _ in range(T)]).flatten()
    ids[ids == 5] = 4  # leave expert 5 empty
    order, offsets = sort_by_expert(ids, E)
    rows = order // k  # token of each sorted (token, slot) pair
    out = grouped_ternary_gemm(x[rows], xs[rows], offsets, packed, scales, K, group, fmt)
    for i, r in enumerate(rows.tolist()):
        e = int(ids[order[i]])
        want = ternary_gemm_ref(x[r:r + 1], xs[r:r + 1], packed[e], scales[e], K, group, fmt)
        assert torch.allclose(out[i:i + 1], want, rtol=1e-5, atol=1e-4)
    with pytest.raises(ValueError, match="offsets"):
        grouped_ternary_gemm(x, xs, [0, T], packed, scales, K, group, fmt)


def test_cuda_backend_absence_is_clean():
    if not torch.cuda.is_available():
        assert cuda_ext() is None
        x, xs = quant_act_int8(torch.randn(2, 64))
        _, packed, scales = _weights(1, 4, 64, 64, "tq2")
        with pytest.raises(RuntimeError, match="unavailable"):
            ternary_gemm(x, xs, packed[0], scales[0], 64, 64, "tq2", backend="cuda")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU unavailable")
@pytest.mark.parametrize("fmt", ["tq1", "tq2"])
@pytest.mark.parametrize("T", [1, 3, 37])
def test_cuda_kernels_match_reference(fmt, T):
    if cuda_ext() is None:
        pytest.skip("CUDA toolkit unavailable")
    E, N, K, group = 5, 70, 4160, 128  # N not a multiple of the 8-row tile; K not of the group
    trits, packed, scales = _weights(E, N, K, group, fmt, seed=7)
    x, xs = quant_act_int8(torch.randn(T, K, generator=torch.Generator().manual_seed(8)))
    ref = ternary_gemm_ref(x, xs, packed[0], scales[0], K, group, fmt)
    got = ternary_gemm(x.cuda(), xs.cuda(), packed[0].cuda(), scales[0].cuda(), K, group, fmt, backend="cuda")
    assert torch.allclose(got.cpu(), ref, rtol=1e-4, atol=1e-3)
    counts = torch.tensor([T, 0, 2 * T, 1, T])
    offsets = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)])
    xg, xsg = quant_act_int8(torch.randn(int(offsets[-1]), K, generator=torch.Generator().manual_seed(9)))
    cpu = grouped_ternary_gemm(xg, xsg, offsets, packed, scales, K, group, fmt, backend="torch")
    gpu = grouped_ternary_gemm(xg.cuda(), xsg.cuda(), offsets, packed.cuda(), scales.cuda(), K, group, fmt,
                               backend="cuda")
    assert torch.allclose(gpu.cpu(), cpu, rtol=1e-4, atol=1e-3)

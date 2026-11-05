"""Ternary GEMM and grouped (MoE expert) GEMM: CUDA kernel, C++ CPU kernel or PyTorch reference.

Contract (shared with ``csrc/boilerplate_cpu.cpp`` and ``csrc/ternary_cuda_kernel.cu``)::

    out[t, n] = x_scale[t] * sum_g w_scale[e(t), n, g] * sum_{k in g} x[t, k] * trit[e(t), n, k]

with int8 activations ``x`` (per-token absmax, ``ternary.ops.quant_act_int8``), packed TQ1/TQ2
trits and float32 group scales. The group sums are exact integers, so every backend agrees up to
float32 rounding of the final scale products.

``grouped_ternary_gemm`` takes the token rows already sorted by expert plus ``offsets`` (expert
``e`` owns rows ``offsets[e]:offsets[e+1]``), the layout a MoE layer gets from one argsort of its
routing. On CUDA it is one launch for all experts; elsewhere it loops over the non-empty experts.

The CUDA kernel is compiled on first use (``torch.utils.cpp_extension``) when a CUDA toolkit is
present; ``BOILERPLATE_NO_CUDA=1`` disables it. It has not been compiled or run on a GPU yet: the
development container has none. ``tests/test_ternary_cuda.py`` compares it with the reference on
any CUDA machine and skips otherwise.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import torch

from ..ternary.pack import _TQ1_LUT, unpack


@lru_cache(1)
def ext():
    if os.environ.get("BOILERPLATE_NO_CUDA") or torch.version.cuda is None or not torch.cuda.is_available():
        return None
    from torch.utils.cpp_extension import CUDA_HOME, load

    if CUDA_HOME is None:
        return None
    src = Path(__file__).parent / "csrc"
    return load(name="boilerplate_cuda_ternary",
                sources=[str(src / "ternary_cuda.cpp"), str(src / "ternary_cuda_kernel.cu")],
                extra_cflags=["-O3"], extra_cuda_cflags=["-O3", "--use_fast_math"], verbose=False)


def _fmt_code(fmt: str) -> int:
    if fmt not in ("tq1", "tq2"):
        raise ValueError(f"unknown ternary packing {fmt!r}")
    return 2 if fmt == "tq2" else 1


def ternary_gemm_ref(x: torch.Tensor, x_scale: torch.Tensor, packed: torch.Tensor, w_scale: torch.Tensor,
                     K: int, group: int, fmt: str) -> torch.Tensor:
    """PyTorch reference: exact integer group sums (float64 matmul of small ints), float32 scaling."""
    t = unpack(packed, fmt, K).to(torch.float64)  # [N, K]
    xs = x.to(torch.float64)
    out = torch.zeros(x.size(0), packed.size(0), dtype=torch.float32, device=x.device)
    for g in range(w_scale.size(1)):
        k0, k1 = g * group, min(K, (g + 1) * group)
        if k0 >= K:
            break
        acc = (xs[:, k0:k1] @ t[:, k0:k1].t()).to(torch.float32)  # exact: |acc| <= 127 * group
        out += acc * w_scale[:, g].float()
    return out * x_scale.float()[:, None]


def ternary_gemm(x: torch.Tensor, x_scale: torch.Tensor, packed: torch.Tensor, w_scale: torch.Tensor,
                 K: int, group: int, fmt: str, backend: str = "auto") -> torch.Tensor:
    """[T, K] int8 activations x [N, Kp] packed weights -> [T, N] float32."""
    code = _fmt_code(fmt)
    if x.dtype != torch.int8:
        raise TypeError("ternary_gemm takes int8 activations (see quant_act_int8)")
    if x.is_cuda and backend in ("auto", "cuda") and ext() is not None:
        return grouped_ternary_gemm(x, x_scale, torch.tensor([0, x.size(0)]), packed[None], w_scale[None],
                                    K, group, fmt, backend="cuda")
    if backend == "cuda":
        raise RuntimeError("CUDA ternary kernel unavailable (no GPU, no toolkit, or BOILERPLATE_NO_CUDA)")
    if not x.is_cuda and backend in ("auto", "cpp"):
        from .cpu import ext as cpu_ext

        e = cpu_ext()
        if e is not None:
            out = torch.zeros(x.size(0), packed.size(0), dtype=torch.float32)
            e.ternary_gemm(out, x.contiguous(), x_scale.float().contiguous(), packed.contiguous(),
                           w_scale.float().contiguous(), K, group, code, _TQ1_LUT)
            return out
    return ternary_gemm_ref(x, x_scale, packed, w_scale, K, group, fmt)


def grouped_ternary_gemm(x: torch.Tensor, x_scale: torch.Tensor, offsets, packed: torch.Tensor,
                         w_scale: torch.Tensor, K: int, group: int, fmt: str, backend: str = "auto") -> torch.Tensor:
    """Rows of ``x`` sorted by expert; ``packed`` [E, N, Kp], ``w_scale`` [E, N, G] -> [T, N] float32."""
    code = _fmt_code(fmt)
    offs = torch.as_tensor(offsets, dtype=torch.int64).cpu()
    E = packed.size(0)
    if offs.numel() != E + 1 or int(offs[0]) != 0 or int(offs[-1]) != x.size(0) or bool((offs.diff() < 0).any()):
        raise ValueError("offsets must be nondecreasing, start at 0, end at the row count, one per expert + 1")
    if x.is_cuda and backend in ("auto", "cuda") and ext() is not None:
        out = torch.zeros(x.size(0), packed.size(1), dtype=torch.float32, device=x.device)
        ext().ternary_grouped_gemm(out, x.contiguous(), x_scale.float().contiguous(), offs.to(x.device),
                                   packed.contiguous(), w_scale.float().contiguous(), K, group, code,
                                   _TQ1_LUT.to(x.device), int(offs.diff().max()) if E else 0)
        return out
    if backend == "cuda":
        raise RuntimeError("CUDA ternary kernel unavailable (no GPU, no toolkit, or BOILERPLATE_NO_CUDA)")
    out = torch.zeros(x.size(0), packed.size(1), dtype=torch.float32, device=x.device)
    for e in range(E):
        a, b = int(offs[e]), int(offs[e + 1])
        if a < b:
            out[a:b] = ternary_gemm(x[a:b], x_scale[a:b], packed[e], w_scale[e], K, group, fmt, backend=backend)
    return out


def sort_by_expert(expert_ids: torch.Tensor, n_experts: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Flattened top-k expert ids [T*k] -> (row order sorting them by expert, offsets [E + 1])."""
    order = torch.argsort(expert_ids, stable=True)
    counts = torch.bincount(expert_ids, minlength=n_experts)
    return order, torch.cat([counts.new_zeros(1), counts.cumsum(0)])

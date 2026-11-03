"""The reference's TileLang ``kernel`` module, replaced by pure-PyTorch equivalents so the official
code runs without a GPU. Implementation: :mod:`boilerplate.kernels.torch_ref`."""

from ..kernels.torch_ref import act_quant, fp4_act_quant, fp4_gemm, fp8_gemm, hc_split_sinkhorn, sparse_attn

__all__ = ["act_quant", "fp4_act_quant", "fp4_gemm", "fp8_gemm", "hc_split_sinkhorn", "sparse_attn"]

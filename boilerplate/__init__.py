"""Boilerplate: a ternary-first inference engine for MoE LLMs."""

import os as _os

# MKL's dynamic threading picks a different number of threads when the machine is busy, which
# changes GEMM summation order: the same run then differs from itself in the last bit, and the
# truncated real-weight model turns that into visibly different logits (see docs/reports/M1.md).
# Fixed thread counts keep CPU runs reproducible. Must be set before torch loads MKL.
_os.environ.setdefault("MKL_DYNAMIC", "FALSE")
_os.environ.setdefault("OMP_DYNAMIC", "FALSE")

__version__ = "0.1.0"

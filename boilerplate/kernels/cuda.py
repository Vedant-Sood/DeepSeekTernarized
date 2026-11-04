"""Optional CUDA Huffman decoder, built on first use when CUDA is available."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import torch


@lru_cache(1)
def ext():
    """Load the CUDA extension, or return None when a CUDA build is unavailable.

    Compilation errors are deliberately not hidden: a machine with a CUDA toolkit must surface
    a broken build rather than silently claiming that the GPU kernel was exercised.
    """
    if os.environ.get("BOILERPLATE_NO_CUDA") or torch.version.cuda is None or not torch.cuda.is_available():
        return None
    from torch.utils.cpp_extension import CUDA_HOME, load

    if CUDA_HOME is None:
        return None
    source_dir = Path(__file__).parent / "csrc"
    return load(
        name="boilerplate_cuda_huffman",
        sources=[str(source_dir / "huffman_cuda.cpp"), str(source_dir / "huffman_cuda_kernel.cu")],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )

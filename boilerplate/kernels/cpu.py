"""C++ CPU kernels (ternary GEMM, Huffman codec), compiled on first use with the system compiler.

``-march=native`` picks AVX2 / AVX-512 on x86; on ARM the portable loops are auto-vectorized for
NEON. Set ``BOILERPLATE_NO_CPP=1`` to force the pure-PyTorch fallbacks.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path


@lru_cache(1)
def ext():
    if os.environ.get("BOILERPLATE_NO_CPP"):
        return None
    from torch.utils.cpp_extension import load

    src = Path(__file__).parent / "csrc" / "boilerplate_cpu.cpp"
    return load(
        name="boilerplate_cpu",
        sources=[str(src)],
        extra_cflags=["-O3", "-march=native", "-fopenmp"],
        extra_ldflags=["-fopenmp"],
        verbose=False,
    )

"""Kernel backends. ``torch_ref``: pure PyTorch, the numerics contract every other backend is
tested against (Mode A on CPU uses it directly). Ternary and entropy-decode kernels arrive in M2/M3."""

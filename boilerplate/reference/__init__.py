"""Reference harness: run a trusted implementation, record per-layer hidden states and logits, and
compare any candidate (e.g. a Boilerplate weight mode) against it."""

from .compare import Comparison, compare_traces
from .trace import ReferenceTrace

__all__ = ["Comparison", "ReferenceTrace", "compare_traces"]

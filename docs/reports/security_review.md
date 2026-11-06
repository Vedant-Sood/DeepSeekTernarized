# Security review

Reviewed against commit `17ee17879fa6c709476fd5dc8243ae450d2f40ef`.
The review covered checkpoint inspection, loading and export; the CPU and CUDA kernels; HTTP
range and metadata reads; training checkpoint serialization; reference imports and evaluation
scripts; and the declared dependencies. Bandit and pip-audit supplemented manual review. This is
a point-in-time review, not a guarantee that every vulnerability has been found.

## Changes

- **Native memory safety:** reject inconsistent ternary matrix dimensions and malformed Huffman
  lengths, lookup tables, offsets and payloads before or during decoding. The CPU decoder reports
  a malformed stream after its parallel section; the CUDA decoder checks its metadata and records
  invalid bitstreams before returning.
- **Training checkpoint deserialization:** write version 2 training checkpoints with the NumPy
  RNG state represented as a tensor and load them using `torch.load(..., weights_only=True)`.
  Version 1 checkpoints used NumPy pickle globals and cannot be loaded by this restricted loader.
  Recreate them or migrate them using separately trusted tooling; there is no automatic unsafe
  pickle fallback. Caller-owned `trainer_state` should contain plain Python data and tensors.
- **Checkpoint files:** reject absolute paths, parent traversal, Windows-style escapes and symlinks
  that resolve outside the checkpoint directory. Validate safetensors header lengths, shapes,
  dtypes, duplicate keys, offsets and file coverage before mapping a local shard. Remote slicing
  also bounds headers and metadata reads and validates URL filenames.
- **Credential handling:** the public-model engram sampling script no longer reads and forwards a
  cached Hugging Face token, and uses the range-validated reader.
- **Build dependencies:** require setuptools 84.1.0 or newer, which includes the fix for
  [GHSA-xmj3-9gf7-h52w](https://github.com/pypa/setuptools/security/advisories/GHSA-xmj3-9gf7-h52w).

## Trust boundaries and remaining risks

- This is a local inference and research tool, not a sandbox or an authenticated multi-user
  service. Model data can request large allocations and substantial compute. Apply operating
  system resource limits and isolation before processing untrusted artifacts.
- Checkpoint directories must not be concurrently modified by an untrusted process between
  validation and use. Hugging Face cache snapshots with symlinks to external blob directories
  must first be copied into a standalone directory.
- The manifest records SHA-256 values, but the loader does not verify them or authenticate the
  source. Obtain and verify checkpoints through a trusted channel.
- Hub references can resolve a mutable default revision. Use a locally downloaded, commit-pinned
  model snapshot when reproducibility or supply-chain verification matters.
- The official reference backend intentionally imports Python from its chosen `code_dir`, and
  the vendored image processor accepts local paths and network URLs. Treat that code directory and
  image records as trusted; do not expose them directly to untrusted users.
- The CUDA changes have regression tests but were not compiled or run in this review; the test
  environment used a CPU-only PyTorch build. CPU native kernels and non-GPU tests were exercised.
- The restricted PyTorch loader reduces the pickle attack surface but does not make arbitrary
  checkpoint contents safe against resource exhaustion. See
  [PyTorch's loading documentation](https://docs.pytorch.org/docs/2.15/generated/torch.load.html).

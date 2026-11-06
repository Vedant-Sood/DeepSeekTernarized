# Security review

## Second review (2026-11-06): whole codebase including the PR #24 branch

Reviewed `main` at `365bc66` plus the then-unmerged `codex/m4-standard-decoder` branch at `2632f0c`
(PR #24: OpenAI-compatible server, QAT training, ternary CUDA GEMM, GGUF export, lm-eval adapter).
Covered: checkpoint and packed-format loading (safetensors, TQ1/TQ2, Huffman, manifests, export
journal), the HTTP server, the CLI and launch scripts, path handling, pickle/`torch.load`, subprocess
and dynamic imports, the C++/CUDA kernels, and declared dependencies. Bandit and pip-audit
supplemented manual review. CPU tests pass; the CUDA kernel was changed but not compiled (no GPU or
CUDA toolkit in the review container), so `tests/test_ternary_cuda.py` must be run on a GPU machine.

| # | Severity | Finding | Status |
|---|----------|---------|--------|
| 1 | High | `serve`: the checkpoint's `chat_template` (from `tokenizer_config.json`) was rendered with a plain `jinja2.Environment`. A crafted template can walk Python object internals (server-side template injection) and run code as the server user. | Fixed: `ImmutableSandboxedEnvironment`, template errors become HTTP 400, only `role`/`content` reach the template. |
| 2 | High | `qat/distributed.py::load_sharded` used `torch.load(..., weights_only=False)` on the per-rank file, so a tampered resume directory runs arbitrary pickle code. This reintroduced the issue PR #23 fixed for single-process checkpoints. | Fixed: version 2 rank files store the NumPy RNG keys as a tensor and load with `weights_only=True`; unknown module names in `levels` are rejected. Version 1 rank files must be recreated. |
| 3 | Medium | `serve` had no authentication, and `scripts/launch/m5_full.sh serve` bound it to `0.0.0.0`, exposing generation (GPU/CPU time) and `/stats` to the whole network. | Fixed: `--api-key` / `BOILERPLATE_API_KEY` (Bearer, constant-time compare; `/health` stays open). Non-loopback binds without a key are refused unless `--insecure-no-auth`. The launch script now defaults to `127.0.0.1` (`HOST=` to override). |
| 4 | Medium | `serve` denial of service: request bodies were read in full with no cap, prompts of any length were tokenized, and any number of requests could queue on the engine lock (each holding a worker thread). | Fixed: 4 MiB body cap (413, checked while streaming), 1 Mi-character prompt cap and 1024 messages (400), 16 queued+running requests (503), uvicorn `limit_concurrency` and a short keep-alive. |
| 5 | Medium | CUDA `ternary_grouped_gemm` trusted `offsets`, `max_rows`, the expert/row dimensions of `w_scale`, and the LUT size, and only checked `Kp`/`G` as lower bounds. A direct call with inconsistent tensors reads and writes GPU memory out of bounds (the Python wrapper validated only `offsets`). | Fixed (not yet compiled): exact shape checks as in the CPU kernel, `offsets` validated on the host, `max_rows` recomputed, `group` bounded so the int32 accumulator cannot overflow, grid limits checked. |
| 6 | Low | Requests longer than the context or KV cache surfaced as HTTP 500 (`RuntimeError: KV cache out of pages`), after spending a prefill. | Fixed: prompt + `max_tokens` checked against `max_seq_len` before generation; KV-cache exhaustion maps to 400. |
| 7 | Low | Parameter type confusion: `true` accepted as `max_tokens`/`n`, non-integer `seed` silently ignored, any iterable accepted as `stop`. | Fixed: strict types, `stop` entries capped at 256 characters. |
| 8 | Low | `m5_full.sh eval` runs lm-eval `humaneval`, which executes model-generated Python on the host once code execution is enabled (`HF_ALLOW_CODE_EVAL=1`, `confirm_run_unsafe_code`). | Open: run code-execution tasks only in a disposable container or VM without credentials or network. |
| 9 | Low | Unpinned Hub revisions: `from_pretrained` calls (`qat/model.py`, `reference/hf_backend.py`, `qat-teacher`, eval scripts) and `huggingface-cli download` in the launch scripts follow the mutable default branch (Bandit B615). None use `trust_remote_code`. | Open: pass `--revision <commit>` in the launch scripts and record it with results. |
| 10 | Low | Local metadata names files without containment checks: `TeacherStore` shard names (`meta.json`), export journal file names, and `cast` copying files that may be symlinks. These are the tool's own outputs. | Open (accepted): treat work directories as trusted; do not resume from directories received from others. |
| 11 | Info | `lm_eval` pulls in `sqlitedict 2.1.0` (PYSEC-2026-1939; pickle-based cache). Only reachable if lm-eval's request cache is enabled on an untrusted database; the adapter does not enable it. | Open: upgrade when a fixed release exists. |

Checked and found sound: all YAML uses `yaml.safe_load`; there is no `subprocess`, `os.system`,
`eval` or `exec` in the package; safetensors/manifest/Mode B/Huffman parsing keeps the PR #23
validation; Python-side TQ1/TQ2 unpacking uses bounds-checked tensor indexing; the CPU ternary and
Huffman kernels validate sizes exactly; the server binds to loopback by default and does not return
tracebacks. The trust-boundary notes below still apply: a checkpoint can request large allocations
and compute, so process untrusted artifacts under OS resource limits.

## First review (PR #23)

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

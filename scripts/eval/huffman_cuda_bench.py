"""Measure the CUDA Huffman kernel on synthetic bytes after checking CPU parity.

This times the kernel into a reused GPU buffer, excluding encoding, host-to-device copies and
decode-table construction. It is not an end-to-end engine benchmark.
"""

import argparse
import json
import os


parser = argparse.ArgumentParser()
parser.add_argument("--symbols", type=int, default=16 << 20)
parser.add_argument("--iterations", type=int, default=20)
parser.add_argument("--warmups", type=int, default=3)
parser.add_argument("--threads", type=int, default=4)
args = parser.parse_args()
if min(args.symbols, args.iterations, args.threads) <= 0 or args.warmups < 0:
    parser.error("symbols, iterations and threads must be positive; warmups must be nonnegative")
os.environ["OMP_NUM_THREADS"] = str(args.threads)
os.environ["MKL_NUM_THREADS"] = str(args.threads)
os.environ["MKL_DYNAMIC"] = "FALSE"

import torch

from boilerplate.format import entropy
from boilerplate.kernels.cuda import ext as cuda_ext

torch.set_num_threads(args.threads)
if not torch.cuda.is_available():
    raise SystemExit("CUDA GPU unavailable")
kernel = cuda_ext()
if kernel is None:
    raise SystemExit("CUDA toolkit unavailable or CUDA extension disabled")

source = torch.randint(0, 16, (args.symbols,), dtype=torch.uint8, generator=torch.Generator().manual_seed(42))
parts = entropy.encode(source)
reference = entropy.decode(parts, args.symbols)
device = torch.device("cuda")
stream = parts["stream"].to(device)
offsets = parts["offsets"].to(device)
lut = entropy.decode_table(parts["lens"], entropy.MAXLEN).to(device)
raw = torch.empty(0, dtype=torch.uint8, device=device)
out = torch.empty(args.symbols, dtype=torch.uint8, device=device)


def decode_once():
    kernel.huff_decode_into(stream, offsets, args.symbols, entropy.BLOCK, lut, entropy.MAXLEN, raw, out)


decode_once()
if not torch.equal(out.cpu(), reference):
    raise RuntimeError("GPU decoder differs from the CPU reference")
for _ in range(args.warmups):
    decode_once()
torch.cuda.synchronize()
start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
start.record()
for _ in range(args.iterations):
    decode_once()
end.record()
end.synchronize()
elapsed_seconds = start.elapsed_time(end) / 1000
print(json.dumps({
    "gpu": torch.cuda.get_device_name(device),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "math_threads": torch.get_num_threads(),
    "omp_num_threads": os.environ["OMP_NUM_THREADS"],
    "symbols": args.symbols,
    "original_bytes_per_iteration": source.numel() * source.element_size(),
    "iterations": args.iterations,
    "elapsed_seconds": elapsed_seconds,
    "decoded_original_gb_per_second": source.numel() * args.iterations / elapsed_seconds / 1e9,
    "scope": "synthetic uint8; kernel only; reused output; excludes H2D, encoding and LUT construction",
    "cpu_gpu_byte_identical": True,
}, indent=2))

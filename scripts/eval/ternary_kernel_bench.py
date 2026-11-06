"""Ternary GEMV/GEMM and grouped expert GEMM timing against BF16 matmul, on CPU or CUDA.

    python scripts/eval/ternary_kernel_bench.py --device cuda --out results/kernel_bench.json

Shapes are the projections of SmolLM2-360M and OLMoE-1B-7B (and an OLMoE layer's 64 experts with
8 of them active per token). For each shape, batch 1 (decode) and 8 and 512 (prefill): time per
call for the ternary kernel (TQ1 and TQ2, int8 activations, quantization included), the
dequantized fallback, and a dense BF16 ``F.linear`` of the same shape, plus the weight bytes read
per call and the implied bandwidth. Results print as JSON with the hardware named.
"""

from __future__ import annotations

import argparse
import json
import platform
import time

import torch
import torch.nn.functional as F

SHAPES = {  # name: (out_features N, in_features K)
    "smollm2.q_proj": (960, 960), "smollm2.gate_proj": (2560, 960), "smollm2.down_proj": (960, 2560),
    "olmoe.q_proj": (2048, 2048), "olmoe.expert.gate_proj": (1024, 2048), "olmoe.expert.down_proj": (2048, 1024),
}


def _timeit(fn, device, reps):
    for _ in range(3):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps


def main(argv=None) -> int:
    from boilerplate.kernels.ternary_cuda import grouped_ternary_gemm, sort_by_expert, ternary_gemm
    from boilerplate.ternary.ops import quant_act_int8
    from boilerplate.ternary.pack import pack

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    dev, g = a.device, torch.Generator().manual_seed(0)
    backend = "cuda" if dev == "cuda" else "auto"
    rows = []
    for name, (N, K) in SHAPES.items():
        trits = torch.randint(-1, 2, (N, K), generator=g).to(torch.int8)
        scale = (torch.rand(N, -(-K // a.group), generator=g) + 0.5).float().to(dev)
        dense = torch.randn(N, K, generator=g).to(dev, torch.bfloat16)
        for fmt in ("tq1", "tq2"):
            packed = pack(trits, fmt).to(dev)
            for M in (1, 8, 512):
                x = torch.randn(M, K, generator=g).to(dev, torch.bfloat16)

                def tern():
                    q, s = quant_act_int8(x)
                    return ternary_gemm(q, s, packed, scale, K, a.group, fmt, backend=backend)

                def deq():
                    q, s = quant_act_int8(x)
                    return ternary_gemm(q, s, packed, scale, K, a.group, fmt, backend="torch")

                reps = a.reps if M < 512 or dev == "cuda" else max(3, a.reps // 10)
                t_t, t_b = _timeit(tern, dev, reps), _timeit(lambda: F.linear(x, dense), dev, reps)
                t_d = _timeit(deq, dev, max(3, reps // 5))
                wbytes = packed.numel() + scale.numel() * 2  # fp16 scales on disk
                rows.append({"shape": name, "N": N, "K": K, "M": M, "fmt": fmt,
                             "ternary_us": t_t * 1e6, "dequant_fallback_us": t_d * 1e6, "bf16_us": t_b * 1e6,
                             "speedup_vs_bf16": t_b / t_t, "ternary_weight_GBps": wbytes / t_t / 1e9,
                             "bf16_weight_GBps": N * K * 2 / t_b / 1e9})
    # grouped experts: one OLMoE MoE projection, 64 experts, top-8, tokens routed at random
    E, k, N, K = 64, 8, 1024, 2048
    trits = torch.randint(-1, 2, (E, N, K), generator=g).to(torch.int8)
    scale = (torch.rand(E, N, K // a.group, generator=g) + 0.5).to(dev)
    for fmt in ("tq1", "tq2"):
        packed = torch.stack([pack(trits[e], fmt) for e in range(E)]).to(dev)
        for T in (1, 8, 512):
            ids = torch.stack([torch.randperm(E, generator=g)[:k] for _ in range(T)]).flatten()
            order, offsets = sort_by_expert(ids, E)
            x = torch.randn(T, K, generator=g).to(dev, torch.bfloat16)
            rows_idx = (order // k).to(dev)

            def grouped():
                q, s = quant_act_int8(x[rows_idx])
                return grouped_ternary_gemm(q, s, offsets, packed, scale, K, a.group, fmt, backend=backend)

            t = _timeit(grouped, dev, a.reps if dev == "cuda" else 5)
            rows.append({"shape": "olmoe.experts.gate_proj (grouped)", "E": E, "top_k": k, "tokens": T, "fmt": fmt,
                         "grouped_us": t * 1e6, "active_experts": int((offsets.diff() > 0).sum())})
    info = {"device": dev, "device_name": torch.cuda.get_device_name() if dev == "cuda" else platform.processor(),
            "torch": torch.__version__, "cuda": torch.version.cuda, "threads": torch.get_num_threads(),
            "group": a.group, "rows": rows}
    text = json.dumps(info, indent=1)
    if a.out:
        open(a.out, "w").write(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

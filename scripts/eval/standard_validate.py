"""Compare standard-decoder hidden states and greedy generation with Hugging Face."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("input_ids", type=Path, help="safetensors file containing [prompts, context] input_ids")
    parser.add_argument("--tokens", type=int, default=64, help="greedy tokens per prompt")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    if args.tokens <= 0 or args.threads <= 0:
        parser.error("tokens and threads must be positive")
    os.environ.update(OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads),
                      MKL_DYNAMIC="FALSE", OMP_DYNAMIC="FALSE", HF_HUB_OFFLINE="1")

    import torch
    import torch.nn.functional as F
    import transformers
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM

    from boilerplate.engine.standard import StandardEngine

    torch.set_num_threads(args.threads)
    ids = load_file(str(args.input_ids))["input_ids"].long()
    if ids.ndim != 2 or not ids.size(0) or not ids.size(1):
        parser.error("input_ids must be a nonempty [batch, context] tensor")
    start = time.perf_counter()
    ref = AutoModelForCausalLM.from_pretrained(args.checkpoint, dtype=torch.float32,
                                               local_files_only=True, attn_implementation="eager").eval()
    engine = StandardEngine.load(args.checkpoint, dtype=torch.float32)
    reference_layers = []
    def capture(_module, _inputs, output):
        reference_layers.append(output[0] if isinstance(output, tuple) else output)
    hooks = [layer.register_forward_hook(capture) for layer in ref.model.layers]
    ref_input = engine_input = ids
    ref_past = engine_past = None
    same_prefix = torch.ones(ids.shape[0], dtype=torch.bool)
    layer_min = None
    layer_max_abs = None
    matches = 0
    try:
        with torch.inference_mode():
            for step in range(args.tokens):
                reference_layers.clear()
                expected = ref(ref_input, past_key_values=ref_past, use_cache=True, output_hidden_states=True)
                actual_layers = []
                actual_logits, _, engine_past = engine.model.forward(engine_input, engine_past, capture_layers=actual_layers)
                ref_past = expected.past_key_values
                if len(reference_layers) != len(actual_layers):
                    raise RuntimeError("reference and engine captured different numbers of decoder layers")
                if layer_min is None:
                    layer_min = [float("inf")] * len(actual_layers)
                    layer_max_abs = [0.0] * len(actual_layers)
                for i, (a, b) in enumerate(zip(reference_layers, actual_layers)):
                    if same_prefix.any():
                        a, b = a[same_prefix].float(), b[same_prefix].float()
                        cosine = F.cosine_similarity(a.flatten(2), b.flatten(2), dim=-1)
                        layer_min[i] = min(layer_min[i], float(cosine.min()))
                        layer_max_abs[i] = max(layer_max_abs[i], float((a - b).abs().max()))
                next_ref = expected.logits[:, -1].argmax(-1)
                next_engine = actual_logits[:, -1].argmax(-1)
                matches += int((next_ref == next_engine).sum())
                same_prefix &= next_ref == next_engine
                ref_input, engine_input = next_ref[:, None], next_engine[:, None]
                if (step + 1) % 8 == 0 or step + 1 == args.tokens:
                    print(f"generated {step + 1}/{args.tokens}; complete matching prefixes: {int(same_prefix.sum())}/{len(same_prefix)}",
                          file=sys.stderr, flush=True)
    finally:
        for hook in hooks:
            hook.remove()
    if torch.get_num_threads() != args.threads:
        raise RuntimeError("PyTorch thread count changed")
    cpu = platform.processor()
    if Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    result = {
        "checkpoint": str(args.checkpoint), "input_ids_sha256": hashlib.sha256(args.input_ids.read_bytes()).hexdigest(),
        "prompts": ids.shape[0], "prompt_tokens": ids.shape[1], "generated_tokens_per_prompt": args.tokens,
        "cpu": cpu, "torch": torch.__version__, "transformers": transformers.__version__,
        "threads": torch.get_num_threads(), "omp_num_threads": os.environ["OMP_NUM_THREADS"],
        "mkl_dynamic": os.environ["MKL_DYNAMIC"], "dtype": "float32", "elapsed_seconds": time.perf_counter() - start,
        "hidden_semantics": "output of each decoder layer before final normalization",
        "layer_min_cosine_same_prefix": layer_min, "layer_max_abs_same_prefix": layer_max_abs,
        "matching_greedy_tokens": matches, "total_greedy_tokens": ids.shape[0] * args.tokens,
        "prompts_with_identical_greedy_tokens": int(same_prefix.sum()),
    }
    print(json.dumps(result, indent=2))
    return 0 if same_prefix.all() and min(layer_min) >= 0.9999 else 1


if __name__ == "__main__":
    raise SystemExit(main())

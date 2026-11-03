"""Reference backend for architectures that HF transformers supports (V2/V3, V4, Llama-like, ...).

Used to validate the harness end-to-end on tiny random-init models today, and as the trusted
reference for any family transformers implements.
"""

from __future__ import annotations

from pathlib import Path

import torch

from .trace import ReferenceTrace


class HFReference:
    def __init__(self, model_path: str | Path, dtype: torch.dtype = torch.float32, device: str = "cpu"):
        from transformers import AutoModelForCausalLM

        self.model_path = str(model_path)
        self.dtype = dtype
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, dtype=dtype).to(device).eval()

    @torch.inference_mode()
    def trace(self, input_ids: torch.Tensor) -> ReferenceTrace:
        out = self.model(input_ids.to(self.model.device), output_hidden_states=True, use_cache=False)
        return ReferenceTrace(
            input_ids=input_ids.cpu(),
            hidden_states=[h.detach().cpu() for h in out.hidden_states],
            logits=out.logits.detach().float().cpu(),
            meta={
                "backend": "hf",
                "model": self.model_path,
                "dtype": str(self.dtype),
                "model_type": self.model.config.model_type,
                "hidden_semantics": "transformers hidden_states: [embeddings, layer_1 ... layer_N (final norm applied to last)]",
            },
        )

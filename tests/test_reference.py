import json
import sys
import textwrap
from pathlib import Path

import pytest
import torch

from boilerplate.checkpoint import CheckpointSource
from boilerplate.config import parse_checkpoint
from boilerplate.reference import ReferenceTrace, compare_traces
from boilerplate.reference.official_backend import OfficialReference, hf_config_to_model_args

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------- config translation


def test_hf_config_maps_to_released_inference_config():
    """Translating the released config.json reproduces the inference/config.json shipped beside it."""
    hf = json.loads((FIXTURES / "deepseek_v41_flash_1017_config.json").read_text())
    expected = json.loads((FIXTURES / "deepseek_v41_flash_1017_inference_config.json").read_text())
    got = hf_config_to_model_args(hf)
    for k, v in expected.items():
        gv = got.get(k)
        assert (list(gv) if isinstance(gv, tuple) else gv) == v, k


# ---------------------------------------------------------------- trace + compare


def _trace(hidden, logits=None, ids=None):
    ids = torch.zeros(1, hidden[0].shape[1], dtype=torch.long) if ids is None else ids
    return ReferenceTrace(ids, hidden, logits, {"backend": "test"})


def test_trace_roundtrip_and_self_compare(tmp_path):
    torch.manual_seed(0)
    hs = [torch.randn(1, 5, 4, 16).bfloat16() for _ in range(3)]
    t = _trace(hs, torch.randn(1, 5, 11))
    t.save(tmp_path / "ref")
    t2 = ReferenceTrace.load(tmp_path / "ref")
    cmp = compare_traces(t, t2)
    assert all(m.bit_exact for m in cmp.layers) and cmp.logits.bit_exact
    assert cmp.logits.kl_mean == 0 and cmp.logits.top1_agreement == 1.0


def test_compare_detects_divergence():
    torch.manual_seed(0)
    hs = [torch.randn(1, 5, 16) for _ in range(2)]
    noisy = [hs[0], hs[1] + 0.5 * torch.randn_like(hs[1])]
    cmp = compare_traces(_trace(hs), _trace(noisy))
    assert cmp.layers[0].bit_exact and cmp.layers[0].cos_min == pytest.approx(1.0)
    assert not cmp.layers[1].bit_exact and cmp.layers[1].cos_mean < 0.99
    with pytest.raises(ValueError):
        compare_traces(_trace(hs), _trace(hs[:1]))


# ---------------------------------------------------------------- HF backend on a tiny random-init V3


@pytest.fixture(scope="module")
def tiny_v3(tmp_path_factory):
    transformers = pytest.importorskip("transformers")
    cfg = transformers.DeepseekV3Config(
        vocab_size=128, hidden_size=64, intermediate_size=128, moe_intermediate_size=32,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=4, n_routed_experts=8,
        n_shared_experts=1, num_experts_per_tok=2, n_group=2, topk_group=1, first_k_dense_replace=1,
        q_lora_rank=32, kv_lora_rank=16, qk_rope_head_dim=8, qk_nope_head_dim=8, v_head_dim=16,
        max_position_embeddings=128,
    )
    torch.manual_seed(0)
    model = transformers.AutoModelForCausalLM.from_config(cfg)
    path = tmp_path_factory.mktemp("v3")
    model.save_pretrained(path)
    return path


def test_parser_on_hf_v3(tiny_v3):
    spec = parse_checkpoint(CheckpointSource.open(tiny_v3))
    assert spec.family == "deepseek_v3" and spec.attention.type == "mla"
    assert spec.attention.kv_lora_rank == 16
    assert spec.moe.moe_layers == [1, 2] and spec.moe.dense_layers == [0]
    assert spec.moe.n_group == 2 and spec.moe.top_k == 2
    assert not [w for w in spec.warnings if "headers" not in w], spec.warnings


def test_hf_reference_trace(tiny_v3, tmp_path):
    from boilerplate.reference.hf_backend import HFReference

    ref = HFReference(tiny_v3)
    ids = torch.randint(0, 128, (2, 9), generator=torch.Generator().manual_seed(0))
    t = ref.trace(ids)
    assert len(t.hidden_states) == 4 and t.hidden_states[0].shape == (2, 9, 64)
    assert t.logits.shape == (2, 9, 128)
    t.save(tmp_path / "v3")
    again = HFReference(tiny_v3).trace(ids)
    cmp = compare_traces(ReferenceTrace.load(tmp_path / "v3"), again)
    assert all(m.bit_exact for m in cmp.layers), cmp.table()


# ---------------------------------------------------------------- official backend plumbing

FAKE_MODEL = '''
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from torch import nn
from kernel import act_quant, fp8_gemm


@dataclass
class ModelArgs:
    max_batch_size: int = 1
    max_seq_len: int = 64
    temperature: float = 1
    dtype: str = "fp8"
    expert_dtype: str = None
    vocab_size: int = 32
    dim: int = 32
    n_layers: int = 2


class Block(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(args.dim, args.dim, dtype=torch.float8_e4m3fn), requires_grad=False)
        self.scale = nn.Parameter(torch.empty(1, 1, dtype=torch.float8_e8m0fnu), requires_grad=False)

    def forward(self, x, start_pos, pre_mix, image_mask):
        q, s = act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu)
        return x + fp8_gemm(q, s, self.weight, self.scale, torch.float8_e8m0fnu, 32), pre_mix


class Head(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(args.vocab_size, args.dim, dtype=torch.float32))

    def forward(self, x, full_logits=False):
        if not full_logits:
            x = x[:, -1]
        return F.linear(x.float(), self.weight)


class Transformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embed = nn.Embedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList([Block(args) for _ in range(args.n_layers)])
        self.head = Head(args)

    @torch.inference_mode()
    def forward(self, input_ids, start_pos=0):
        h = self.embed(input_ids)
        for layer in self.layers:
            h, _ = layer(h, start_pos, None, None)
        logits = self.head(h)
        return logits.argmax(-1), logits, None
'''

FAKE_CONVERT = '''
import os
from glob import glob
from safetensors.torch import load_file, save_file

def main(hf_ckpt_path, save_path, mp, expert_dtype, tokenizer_path=None):
    sd = {}
    for f in glob(os.path.join(hf_ckpt_path, "*.safetensors")):
        sd.update(load_file(f))
    save_file(sd, os.path.join(save_path, f"model0-mp{mp}.safetensors"))
'''


def test_official_backend_with_cpu_kernel_shim(tmp_path):
    from safetensors.torch import save_file

    from boilerplate.checkpoint.dequant import quant_fp8_block

    code = tmp_path / "inference"
    code.mkdir()
    (code / "model.py").write_text(textwrap.dedent(FAKE_MODEL))
    (code / "convert.py").write_text(textwrap.dedent(FAKE_CONVERT))
    # no kernel.py here: the import only succeeds if the CPU shim is injected
    ckpt = tmp_path / "hf"
    ckpt.mkdir()
    (ckpt / "config.json").write_text(json.dumps({"text_config": {"vocab_size": 32, "hidden_size": 32, "num_hidden_layers": 2}}))
    torch.manual_seed(0)
    sd = {"embed.weight": torch.randn(32, 32).bfloat16(), "head.weight": torch.randn(32, 32)}
    for i in range(2):
        w, s = quant_fp8_block(torch.randn(32, 32) / 8, (32, 32))
        sd[f"layers.{i}.weight"], sd[f"layers.{i}.scale"] = w, s
    save_file(sd, str(ckpt / "model.safetensors"))

    ref = OfficialReference(code, ckpt, tmp_path / "work", use_cpu_kernels=True)
    assert ref.args_kwargs["dim"] == 32 and ref.args_kwargs["n_layers"] == 2
    ids = torch.randint(0, 32, (1, 7))
    t = ref.trace(ids)
    assert len(t.hidden_states) == 2 and t.hidden_states[1].shape == (1, 7, 32)
    assert t.logits.shape == (1, 7, 32)  # full-sequence logits, not last position only
    assert "kernel" not in sys.modules and "model" not in sys.modules  # no leakage
    assert torch.get_default_dtype() == torch.float32
    assert torch.allclose(t.logits[0, -1], ref.trace(ids).logits[0, -1])

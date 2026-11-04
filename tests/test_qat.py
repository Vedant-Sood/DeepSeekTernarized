"""QAT primitive checks on tiny synthetic tensors; no pretrained model is trained."""

import pytest
import torch
from torch.nn import functional as F

from boilerplate.qat import BitLinear, absmean_ternary, warmup_level
from boilerplate.ternary.ops import TernaryLinear, quant_act_int8


def test_absmean_ternary_matches_reference_including_short_and_zero_groups():
    weight = torch.tensor([[0.0, -2.0, 4.0, 3.0, -1.0], [0.0, 0.0, 0.0, 0.0, 0.0]])
    result = absmean_ternary(weight, group=3)
    expected_scales = torch.tensor([[2.0, 2.0], [0.0, 0.0]], dtype=torch.float16)
    expected_trits = torch.tensor([[0, -1, 1, 1, 0], [0, 0, 0, 0, 0]], dtype=torch.int8)
    assert torch.equal(result.scale, expected_scales)
    assert torch.equal(result.trits, expected_trits)


def test_warmup_and_full_precision_endpoint():
    assert [warmup_level(i, 4) for i in (0, 1, 4, 9)] == [0.0, 0.25, 1.0, 1.0]
    assert warmup_level(0, 0) == 1.0
    with pytest.raises(ValueError):
        warmup_level(-1, 4)
    model = BitLinear(7, 5, bias=True)
    x = torch.randn(2, 7, dtype=torch.bfloat16)
    assert model.weight.dtype == torch.bfloat16
    assert torch.equal(model(x), F.linear(x, model.weight, model.bias))
    model.set_quantization_level(0.25)
    restored = BitLinear(7, 5, bias=True)
    restored.load_state_dict(model.state_dict())
    assert restored.level == 0.25
    assert torch.equal(restored(x), model(x))


def test_ste_gradients_match_identity_estimator():
    torch.manual_seed(21)
    model = BitLinear(7, 4, bias=True, group=3, master_dtype=torch.float32)
    model.set_quantization_level(1)
    x = torch.randn(3, 7, requires_grad=True)
    upstream = torch.randn(3, 4)
    model(x).backward(upstream)
    term = absmean_ternary(model.weight.detach(), 3)
    quant_w = term.trits.float() * term.scale.float().repeat_interleave(3, -1)[:, :7]
    qx, sx = quant_act_int8(x.detach())
    quant_x = qx.float() * sx.unsqueeze(-1)
    assert torch.allclose(x.grad, upstream @ quant_w, atol=1e-6, rtol=1e-6)
    assert torch.allclose(model.weight.grad, upstream.T @ quant_x, atol=1e-6, rtol=1e-6)
    assert torch.allclose(model.bias.grad, upstream.sum(0), atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("fmt", ["tq1", "tq2"])
def test_training_graph_matches_packed_engine(fmt):
    torch.manual_seed(22)
    model = BitLinear(131, 9, group=64)
    model.set_quantization_level(1)
    x = torch.randn(2, 3, 131, dtype=torch.bfloat16)
    expected = TernaryLinear([model.packed_term(fmt)], 131, fmt, 64, act="int8", backend="torch")(x)
    assert torch.allclose(model(x).float(), expected.float(), atol=1e-3, rtol=1e-3)

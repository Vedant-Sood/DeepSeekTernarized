"""Routing-drift metrics, BitLinear conversion and the 2-process gloo QAT dry run."""

import pytest
import torch

from boilerplate.qat.routing import RoutingDrift, capture_hf_routing, freeze_routers, load_entropy, topk_overlap


def test_topk_overlap_and_load_entropy():
    ref = torch.tensor([[0, 1], [2, 3], [0, 2]])
    assert topk_overlap(ref, ref) == 1.0
    assert topk_overlap(ref, torch.tensor([[1, 0], [3, 1], [1, 3]])) == pytest.approx((1 + 0.5 + 0) / 3)
    assert load_entropy(torch.tensor([[0, 1], [2, 3]]), 4) == pytest.approx(1.0)
    assert load_entropy(torch.tensor([[1, 1], [1, 1]]), 4) == 0.0
    with pytest.raises(ValueError):
        topk_overlap(ref, ref[:2])
    drift = RoutingDrift(2, 4)
    drift.update([ref, ref], [ref, ref.flip(-1)])
    summary = drift.summary()
    assert [s["topk_overlap"] for s in summary] == [1.0, 1.0]


def test_hf_router_capture_and_freeze():
    from boilerplate.qat.smoke import tiny_model

    pytest.importorskip("transformers")
    model = tiny_model("olmoe")
    assert freeze_routers(model) == model.config.num_hidden_layers
    assert all(not p.requires_grad for n, p in model.named_parameters() if ".mlp.gate." in n)
    ids = torch.randint(0, 64, (2, 5), generator=torch.Generator().manual_seed(3))
    with torch.no_grad(), capture_hf_routing(model, model.config.num_experts_per_tok) as calls:
        model(ids)
    assert len(calls) == 1 and len(calls[0]) == model.config.num_hidden_layers
    assert calls[0][0].shape == (10, model.config.num_experts_per_tok)


def test_convert_linears_keeps_function_at_level_zero():
    from boilerplate.qat import BitLinear
    from boilerplate.qat.distributed import convert_linears
    from boilerplate.qat.smoke import tiny_model, unfuse_experts

    pytest.importorskip("transformers")
    model = tiny_model("olmoe").eval()
    ids = torch.randint(0, 64, (1, 7), generator=torch.Generator().manual_seed(4))
    with torch.no_grad():
        expected = model(ids).logits
    unfuse_experts(model)
    names = convert_linears(model, group=16, master_dtype=torch.float32)
    assert names and all(isinstance(model.get_submodule(n), BitLinear) for n in names)
    assert not any("gate" == n.rsplit(".", 1)[-1] or "lm_head" in n or "embed" in n for n in names)
    with torch.no_grad():
        actual = model(ids).logits
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("kind", ["llama", "olmoe"])
def test_two_process_gloo_qat_resume_is_exact(tmp_path, kind):
    pytest.importorskip("transformers")
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        pytest.skip("gloo backend unavailable")
    from boilerplate.qat.smoke import run

    result = run(world=2, kind=kind, steps=3, warmup=2, out=str(tmp_path))
    assert result["resume_exact"], result
    assert all(r["start"] == 2 for r in result["resumes"])
    # the two ranks see different data but share one (sharded) model: losses differ per rank
    assert result["runs"][0]["losses"] != result["runs"][1]["losses"]
    if kind == "olmoe":
        assert all(r["frozen_routers"] == 2 for r in result["runs"])
        for layer in result["runs"][0]["routing"]:
            assert 0.0 <= layer["topk_overlap"] <= 1.0

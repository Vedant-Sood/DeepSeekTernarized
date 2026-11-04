"""Resume a tiny random model without training any pretrained checkpoint."""

import random

import numpy as np
import pytest
import torch

from boilerplate.qat import BitLinear
from boilerplate.qat.checkpoint import load_checkpoint, save_checkpoint


def _step(model, optimizer, generator):
    x = torch.randn(3, 7) + random.random() + float(np.random.random())
    target = torch.randn(3, 4, generator=generator)
    loss = (model(x) - target).square().mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return loss.detach()


def test_resume_restores_parameters_optimizer_rng_and_warmup(tmp_path):
    py_rng, np_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    try:
        random.seed(8)
        np.random.seed(9)
        torch.manual_seed(10)
        generator = torch.Generator().manual_seed(11)
        model = BitLinear(7, 4, group=3, master_dtype=torch.float32)
        model.set_quantization_level(0.75)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        _step(model, optimizer, generator)
        _step(model, optimizer, generator)
        path = tmp_path / "checkpoint.pt"
        save_checkpoint(path, model, optimizer, step=2, tokens_seen=6,
                        trainer_state={"sampler_position": 6}, generator=generator)
        expected_loss = _step(model, optimizer, generator)

        restored = BitLinear(7, 4, group=3, master_dtype=torch.float32)
        restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.001)
        restored_generator = torch.Generator()
        progress = load_checkpoint(path, restored, restored_optimizer, generator=restored_generator)
        actual_loss = _step(restored, restored_optimizer, restored_generator)
        assert progress == {"step": 2, "tokens_seen": 6, "trainer_state": {"sampler_position": 6}}
        assert restored.level == 0.75
        assert torch.equal(expected_loss, actual_loss)
        assert all(torch.equal(a, b) for a, b in zip(model.parameters(), restored.parameters()))
        for left, right in zip(optimizer.state.values(), restored_optimizer.state.values()):
            assert left.keys() == right.keys()
            for key in left:
                assert torch.equal(left[key], right[key])
        with pytest.raises(ValueError, match="generator"):
            load_checkpoint(path, restored, restored_optimizer)
    finally:
        random.setstate(py_rng)
        np.random.set_state(np_rng)
        torch.set_rng_state(torch_rng)

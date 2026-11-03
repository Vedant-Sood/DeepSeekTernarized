import torch

from scripts.eval.modeb_validate import same_bits


def test_bit_comparison_detects_different_signed_zero():
    positive = torch.tensor([0.0], dtype=torch.float32)
    negative = torch.tensor([-0.0], dtype=torch.float32)
    assert torch.equal(positive, negative)
    assert not same_bits(positive, negative)
    assert same_bits(positive, positive.clone())

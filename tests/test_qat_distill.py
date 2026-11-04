import torch
from torch.nn import functional as F

from boilerplate.qat.distill import distillation_loss, teacher_topk, topk_teacher_kl


def test_topk_tail_bucket_matches_explicit_coarsened_kl():
    torch.manual_seed(31)
    teacher_logits, student_logits = torch.randn(2, 3, 7), torch.randn(2, 3, 7, requires_grad=True)
    temperature = 1.7
    teacher = teacher_topk(teacher_logits, 3, temperature)
    p = F.softmax(teacher_logits / temperature, -1)
    q = F.softmax(student_logits / temperature, -1)
    top_p, top_q = p.gather(-1, teacher["indices"]), q.gather(-1, teacher["indices"])
    tail_p, tail_q = 1 - top_p.sum(-1), 1 - top_q.sum(-1)
    expected = ((top_p * (top_p.log() - top_q.log())).sum(-1) + tail_p * (tail_p.log() - tail_q.log())) * temperature**2
    actual = topk_teacher_kl(student_logits, teacher)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
    actual.mean().backward()
    assert torch.isfinite(student_logits.grad).all()


def test_full_vocabulary_kl_and_masked_ce():
    torch.manual_seed(32)
    teacher_logits, student_logits = torch.randn(2, 3, 7), torch.randn(2, 3, 7, requires_grad=True)
    teacher = teacher_topk(teacher_logits, 7, 2.0)
    expected_kl = F.kl_div(F.log_softmax(student_logits / 2, -1), F.log_softmax(teacher_logits / 2, -1),
                           log_target=True, reduction="none").sum(-1) * 4
    assert torch.allclose(topk_teacher_kl(student_logits, teacher), expected_kl, atol=1e-6, rtol=1e-6)
    teacher["indices"] = teacher["indices"].to(torch.int32)
    assert torch.allclose(topk_teacher_kl(student_logits, teacher), expected_kl, atol=1e-6, rtol=1e-6)
    labels = torch.tensor([[1, 2, -100], [3, -100, 4]])
    ce = F.cross_entropy(student_logits.reshape(-1, 7), labels.reshape(-1))
    expected = 0.7 * ce + 0.3 * expected_kl[labels != -100].mean()
    loss = distillation_loss(student_logits, labels, teacher, kl_weight=0.3)
    assert torch.allclose(loss, expected, atol=1e-6, rtol=1e-6)
    loss.backward()
    assert torch.isfinite(student_logits.grad).all()

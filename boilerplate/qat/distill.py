"""Distillation with offline teacher top-k probabilities and an aggregate tail bucket."""

from __future__ import annotations

import torch
from torch.nn import functional as F


@torch.no_grad()
def teacher_topk(logits: torch.Tensor, k: int, temperature: float = 1.0) -> dict[str, torch.Tensor]:
    """Return tensors that can be written to safetensors without retaining the teacher.

    Top-k entries retain their normalized log probabilities; all remaining vocabulary entries
    share one aggregate tail bucket. This is a coarsened distribution when k < vocabulary size.
    """
    if logits.ndim < 1 or not 0 < k <= logits.shape[-1] or temperature <= 0:
        raise ValueError("k must be in 1..vocabulary and temperature must be positive")
    log_probs = F.log_softmax(logits.float() / temperature, dim=-1)
    values, indices = log_probs.topk(k, dim=-1)
    if k == logits.shape[-1]:
        tail = torch.full_like(values[..., 0], -torch.inf)
    else:
        keep = torch.ones_like(log_probs, dtype=torch.bool).scatter_(-1, indices, False)
        tail = torch.logsumexp(log_probs.masked_fill(~keep, -torch.inf), dim=-1)
    return {"indices": indices, "log_probs": values, "tail_log_prob": tail,
            "temperature": torch.tensor(temperature, dtype=torch.float32, device=logits.device)}


def topk_teacher_kl(student_logits: torch.Tensor, teacher: dict[str, torch.Tensor]) -> torch.Tensor:
    """KL per token against the stored teacher top-k plus tail, multiplied by temperature²."""
    indices = teacher["indices"].to(device=student_logits.device, dtype=torch.long)
    target = teacher["log_probs"].to(device=student_logits.device, dtype=torch.float32)
    target_tail = teacher["tail_log_prob"].to(device=student_logits.device, dtype=torch.float32)
    temperature = float(teacher["temperature"])
    if (temperature <= 0 or indices.shape != target.shape or indices.shape[:-1] != student_logits.shape[:-1]
            or target_tail.shape != student_logits.shape[:-1] or not 0 < indices.shape[-1] <= student_logits.shape[-1]):
        raise ValueError("teacher distribution shape or temperature is invalid")
    student = F.log_softmax(student_logits.float() / temperature, dim=-1)
    top = student.gather(-1, indices)
    kl = (target.exp() * (target - top)).sum(-1)
    if indices.shape[-1] < student_logits.shape[-1]:
        tail_mask = torch.ones_like(student, dtype=torch.bool).scatter_(-1, indices, False)
        student_tail = torch.logsumexp(student.masked_fill(~tail_mask, -torch.inf), dim=-1)
        kl = kl + target_tail.exp() * (target_tail - student_tail)
    return kl * temperature**2


def distillation_loss(student_logits: torch.Tensor, labels: torch.Tensor, teacher: dict[str, torch.Tensor],
                      *, kl_weight: float = 0.5, ignore_index: int = -100) -> torch.Tensor:
    """Blend next-token CE and teacher KL for already-aligned logits and labels."""
    if not 0 <= kl_weight <= 1 or student_logits.shape[:-1] != labels.shape:
        raise ValueError("labels must match the logits and kl_weight must be in [0, 1]")
    valid = labels != ignore_index
    if not bool(valid.any()):
        raise ValueError("at least one label must be valid")
    ce = F.cross_entropy(student_logits.float().reshape(-1, student_logits.shape[-1]), labels.reshape(-1),
                         ignore_index=ignore_index)
    kl = topk_teacher_kl(student_logits, teacher)[valid].mean()
    return (1 - kl_weight) * ce + kl_weight * kl

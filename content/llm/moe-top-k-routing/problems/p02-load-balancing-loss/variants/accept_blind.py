from __future__ import annotations

import math

import torch


def load_balance_loss(logits: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """Switch-style auxiliary load-balancing loss, loss = alpha * E * sum_i f_i * P_i."""
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise TypeError("alpha must be a non-negative finite int or float")
    if not isinstance(logits, torch.Tensor):
        raise TypeError("logits must be a torch.Tensor")
    if logits.dim() != 2:
        raise ValueError("logits must have shape (T, E)")
    if logits.dtype != torch.float64:
        raise ValueError("logits must be float64")
    if not torch.isfinite(logits).all():
        raise ValueError("logits must be finite")
    num_tokens, num_experts = logits.shape
    if num_tokens < 1:
        raise ValueError("there must be at least one token")
    if num_experts < 1:
        raise ValueError("there must be at least one expert")
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and non-negative")

    probs = torch.softmax(logits, dim=1)
    mean_prob = probs.mean(dim=0)  # P_i, carries the gradient

    # Top-1 expert per token, ties to the lower index (stable sort of negated logits).
    top1 = torch.sort(-logits, dim=1, stable=True).indices[:, 0]
    counts = torch.zeros(num_experts, dtype=torch.float64)
    counts.scatter_add_(0, top1, torch.ones(num_tokens, dtype=torch.float64))
    fraction = counts / num_tokens  # f_i, a constant

    return alpha * num_experts * torch.sum(fraction * mean_prob)

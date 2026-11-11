from __future__ import annotations

import math

import torch


def load_balance_loss(logits: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """Switch-style load-balancing loss, loss = alpha * E * sum_i f_i * P_i (form from ST-MoE p.31).

    Shapes: logits (T, E). P (E,) is the mean router probability per expert. f (E,) is the
    fraction of tokens dispatched to each expert (top-1). The result is a 0-dim tensor.
    """
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise TypeError("alpha must be a real number")
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and non-negative")
    if logits.ndim != 2:
        raise ValueError("logits must be 2-D (T, E)")
    if logits.dtype != torch.float64:
        raise ValueError("logits must be float64")
    T, E = logits.shape
    if T == 0 or E == 0:
        raise ValueError("need at least one token and one expert")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("logits must be finite")
    probs = torch.softmax(logits, dim=-1)
    P = probs.mean(dim=0)
    top1 = torch.sort(logits, dim=-1, descending=True, stable=False).indices[:, 0]
    f = torch.bincount(top1, minlength=E).to(torch.float64) / T
    return alpha * E * (f * P).sum()

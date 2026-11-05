from __future__ import annotations

import math

import torch


def load_balance_loss(logits: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """Switch-style load-balancing loss, loss = alpha * E * sum_i f_i * P_i (form from ST-MoE p.31).

    Shapes: logits (T, E). P (E,) is the mean router probability per expert. f (E,) is the
    fraction of tokens dispatched to each expert (top-1). The result is a 0-dim tensor.
    """
    # Validate the interface first. bool is an int in Python, so it is rejected by name.
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

    # Step 1, router probabilities. Softmax over the expert axis, one row per token: p[t, i].
    probs = torch.softmax(logits, dim=-1)  # (T, E), differentiable

    # Step 2, P_i = (1/T) sum_t p[t, i]. The mean over tokens is the differentiable part of the
    # loss. Gradients reach the logits through this line only.
    P = probs.mean(dim=0)  # (E,)

    # Step 3, f_i = (1/T) sum_t 1[top-1 of token t is i]. The top-1 expert is the first column of
    # a stable descending sort, so a tie goes to the lower index. Counting is not differentiable,
    # so f carries no gradient.
    top1 = torch.sort(logits, dim=-1, descending=True, stable=True).indices[:, 0]  # (T,)
    f = torch.bincount(top1, minlength=E).to(torch.float64) / T  # (E,)

    # Step 4, the loss. The factor E makes a uniform router score exactly alpha for any E.
    return alpha * E * (f * P).sum()

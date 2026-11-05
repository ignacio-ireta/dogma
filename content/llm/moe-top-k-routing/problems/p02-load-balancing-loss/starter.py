from __future__ import annotations

import torch


def load_balance_loss(logits: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """Switch-style auxiliary load-balancing loss, loss = alpha * E * sum_i f_i * P_i.

    Shapes: logits (T, E) float64, finite, T >= 1, E >= 1. alpha is a non-negative finite
    int or float (bool is rejected). P_i = mean over tokens of the softmax probability of
    expert i. f_i = fraction of tokens whose top-1 expert is i, ties to the lower index.
    f is a constant, so gradients flow through P only. Must not mutate logits.

    Returns a 0-dimensional float64 tensor.
    Raises TypeError for a bool or non-numeric alpha, and ValueError for a wrong rank or
    dtype, a non-finite logit, T = 0, E = 0, or a negative or non-finite alpha.
    """
    raise NotImplementedError

from __future__ import annotations

import torch


def batch_norm_backward(
    grad_y: torch.Tensor,
    x: torch.Tensor,
    gamma: torch.Tensor,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (grad_x, grad_gamma, grad_beta) for training-mode batch norm of an (N, C) batch.

    Recompute the per-feature batch mean and biased variance from x, then apply the closed-form
    backward pass for y = gamma * (x - mean) / sqrt(var + eps) + beta. Raise ValueError for
    invalid shapes or dtypes, N < 2, gamma not of shape (C,), or eps <= 0. Never mutate inputs.
    """
    raise NotImplementedError

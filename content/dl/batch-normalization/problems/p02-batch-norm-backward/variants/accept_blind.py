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
    if x.dim() != 2 or grad_y.dim() != 2:
        raise ValueError("x and grad_y must be 2-D")
    if not (x.is_floating_point() and grad_y.is_floating_point()):
        raise ValueError("x and grad_y must be floating-point")
    if grad_y.shape != x.shape:
        raise ValueError("grad_y and x must have the same shape")
    n, c = x.shape
    if n < 2:
        raise ValueError("batch norm backward needs at least two rows")
    if c < 1:
        raise ValueError("x must have at least one feature")
    if gamma.shape != (c,):
        raise ValueError("gamma must have shape (C,)")
    if not eps > 0:
        raise ValueError("eps must be positive")

    mu = x.mean(dim=0)
    xc = x - mu
    var = (xc * xc).mean(dim=0)
    inv_std = 1.0 / torch.sqrt(var + eps)
    x_hat = xc * inv_std

    grad_beta = grad_y.sum(dim=0)
    grad_gamma = (grad_y * x_hat).sum(dim=0)

    dxhat = grad_y * gamma
    sum_dxhat = dxhat.sum(dim=0)
    sum_dxhat_xhat = (dxhat * x_hat).sum(dim=0)
    grad_x = (inv_std / n) * (n * dxhat - sum_dxhat - x_hat * sum_dxhat_xhat)

    return grad_x.to(x.dtype), grad_gamma.to(x.dtype), grad_beta.to(x.dtype)

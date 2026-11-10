from __future__ import annotations

import torch


def batch_norm_backward(
    grad_y: torch.Tensor,
    x: torch.Tensor,
    gamma: torch.Tensor,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Wrong on purpose: leaves out the variance path in grad_x. Return (grad_x, grad_gamma, grad_beta) for training-mode batch norm of an (N, C) batch."""  # noqa: E501
    if x.ndim != 2 or not x.is_floating_point():
        raise ValueError("x must be a floating-point tensor of shape (N, C)")
    if grad_y.shape != x.shape or not grad_y.is_floating_point():
        raise ValueError("grad_y must be a floating-point tensor with the shape of x")
    n_rows, n_features = x.shape
    if n_rows < 2:
        raise ValueError("training-mode batch norm backward needs at least two rows")
    if gamma.shape != (n_features,):
        raise ValueError("gamma must have shape (C,) matching the feature count")
    if eps <= 0.0:
        raise ValueError("eps must be positive")

    # Recompute the forward quantities used by the closed form (biased variance, divide by N).
    mean = x.mean(dim=0)
    centred = x - mean
    sigma = torch.sqrt((centred * centred).mean(dim=0) + eps)
    x_hat = centred / sigma

    # Parameter gradients: each parameter is shared by every row, so its gradient sums over rows.
    grad_beta = grad_y.sum(dim=0)  # dL/dbeta_c = sum_i g_ic
    grad_gamma = (grad_y * x_hat).sum(dim=0)  # dL/dgamma_c = sum_i g_ic * x_hat_ic

    # Input gradient. Every row affects the batch mean and variance, so the gradient collects
    # three terms: the direct path, the mean path (sum_i g_i) and the variance path (x_hat term).
    # dx = gamma / (N * sigma) * (N * g - sum_i g_i - x_hat * sum_i g_i * x_hat_i)
    scale = gamma / (n_rows * sigma)
    grad_x = scale * (n_rows * grad_y - grad_y.sum(dim=0))
    return grad_x, grad_gamma, grad_beta

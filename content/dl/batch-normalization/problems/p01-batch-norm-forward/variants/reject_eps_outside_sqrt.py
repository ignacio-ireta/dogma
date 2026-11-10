from __future__ import annotations

import torch


def _validate(
    x: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    *,
    training: bool,
    momentum: float,
    eps: float,
) -> None:
    """Reject inputs outside the contract before any arithmetic runs."""
    if x.ndim != 2 or not x.is_floating_point():
        raise ValueError("x must be a floating-point tensor of shape (N, C)")
    n_rows, n_features = x.shape
    if n_rows == 0 or n_features == 0:
        raise ValueError("x must have at least one row and one feature")
    if training and n_rows < 2:
        # With one row every batch variance is zero, so the statistics carry no information.
        raise ValueError("training-mode batch statistics need at least two rows")
    for name, tensor in (
        ("gamma", gamma),
        ("beta", beta),
        ("running_mean", running_mean),
        ("running_var", running_var),
    ):
        if tensor.shape != (n_features,):
            raise ValueError(f"{name} must have shape (C,) matching the feature count")
    if not 0.0 <= momentum <= 1.0:
        raise ValueError("momentum must lie in [0, 1]")
    if eps <= 0.0:
        raise ValueError("eps must be positive")


def batch_norm_forward(
    x: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    *,
    training: bool,
    momentum: float = 0.1,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Wrong on purpose: eps is added to the standard deviation, not inside the square root. Batch-normalize an (N, C) batch per feature; return (y, new_running_mean, new_running_var).

    Inputs are never mutated: the running statistics are returned as new tensors.
    """  # noqa: E501
    _validate(
        x,
        gamma,
        beta,
        running_mean,
        running_var,
        training=training,
        momentum=momentum,
        eps=eps,
    )
    if training:
        n_rows = x.shape[0]
        # Step 1: per-feature batch mean over the N rows (axis 0). Shape (C,).
        batch_mean = x.mean(dim=0)
        centred = x - batch_mean
        # Step 2: normalize with the biased variance (divide by N), so each batch is
        # standardized exactly. Centering before squaring avoids cancellation.
        batch_var_biased = (centred * centred).mean(dim=0)
        normalized = centred / (torch.sqrt(batch_var_biased) + eps)
        # Step 3: the running estimate tracks the population variance, so it uses the
        # unbiased variance: multiply the biased value by N / (N - 1).
        batch_var_unbiased = batch_var_biased * n_rows / (n_rows - 1)
        # Step 4: exponential moving average. Momentum weights the NEW batch statistic
        # (PyTorch convention, default 0.1).
        new_mean = (1.0 - momentum) * running_mean + momentum * batch_mean
        new_var = (1.0 - momentum) * running_var + momentum * batch_var_unbiased
    else:
        # Inference: the stored statistics are the normalizer; they do not change.
        normalized = (x - running_mean) / torch.sqrt(running_var + eps)
        new_mean = running_mean.clone()
        new_var = running_var.clone()
    # Step 5: learned affine transform, applied per feature.
    y = gamma * normalized + beta
    return y, new_mean, new_var

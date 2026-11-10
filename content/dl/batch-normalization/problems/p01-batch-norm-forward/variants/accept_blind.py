from __future__ import annotations

import torch


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
    """Batch-normalize an (N, C) batch per feature; return (y, new_running_mean, new_running_var).

    Training: normalize with biased batch variance (divide by N); update the running statistics
    with the batch mean and the unbiased variance (divide by N - 1), where momentum weights the
    new batch statistic. Inference: normalize with the running statistics and return them
    unchanged. Raise ValueError for invalid shapes, dtypes, N == 0, training with N < 2,
    momentum outside [0, 1], or eps <= 0. Never mutate any input tensor.
    """
    if not isinstance(x, torch.Tensor) or x.ndim != 2 or not x.is_floating_point():
        raise ValueError("x must be a 2-D floating-point tensor")
    n, c = x.shape
    if n == 0 or c == 0:
        raise ValueError("x must have at least one row and one feature")
    for name, t in (
        ("gamma", gamma),
        ("beta", beta),
        ("running_mean", running_mean),
        ("running_var", running_var),
    ):
        if not isinstance(t, torch.Tensor) or t.ndim != 1 or t.shape[0] != c:
            raise ValueError(f"{name} must be a 1-D tensor of length C")
    if training and n < 2:
        raise ValueError("training mode needs at least two rows for the batch variance")
    if not (0.0 <= momentum <= 1.0):
        raise ValueError("momentum must lie in [0, 1]")
    if not (eps > 0):
        raise ValueError("eps must be positive")

    if training:
        mean = x.mean(dim=0)
        centered = x - mean
        biased_var = (centered * centered).mean(dim=0)
        unbiased_var = biased_var * (n / (n - 1))
        y = centered / torch.sqrt(biased_var + eps) * gamma + beta
        new_mean = (1.0 - momentum) * running_mean + momentum * mean.to(running_mean.dtype)
        new_var = (1.0 - momentum) * running_var + momentum * unbiased_var.to(running_var.dtype)
    else:
        y = (x - running_mean) / torch.sqrt(running_var + eps) * gamma + beta
        new_mean = running_mean.clone()
        new_var = running_var.clone()

    return y.to(x.dtype), new_mean, new_var

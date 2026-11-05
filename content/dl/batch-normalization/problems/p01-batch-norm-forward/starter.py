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
    raise NotImplementedError

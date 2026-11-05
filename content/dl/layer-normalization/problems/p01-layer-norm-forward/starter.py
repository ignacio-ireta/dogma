from __future__ import annotations

import torch


def layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Normalize x over its last dimension, then apply per-feature weight and bias.

    x has shape (..., D) and is floating point; weight and bias have shape (D,). Statistics are the
    mean and the biased variance (divide by D) over the last dimension, centred before squaring,
    with eps added inside the square root. Raise ValueError for a 0-dimensional or non-floating x,
    D == 0, weight or bias not of shape (D,), or eps <= 0. Never mutate inputs.
    """
    raise NotImplementedError

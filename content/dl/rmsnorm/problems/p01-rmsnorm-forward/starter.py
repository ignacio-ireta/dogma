from __future__ import annotations

import torch


def rms_norm(
    x: torch.Tensor,
    gain: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Scale each length-D vector of x by its root mean square, then by the per-feature gain.

    x has shape (..., D) and is floating point; gain has shape (D,). Do not subtract the mean.
    The statistic is mean(x**2) over the last dimension, with eps added inside the square root.
    Raise ValueError for a 0-dimensional or non-floating x, D == 0, gain not of shape (D,), or
    eps <= 0. Never mutate inputs.
    """
    raise NotImplementedError

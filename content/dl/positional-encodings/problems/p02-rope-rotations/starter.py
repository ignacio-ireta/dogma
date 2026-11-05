from __future__ import annotations

import torch


def apply_rope(x: torch.Tensor) -> torch.Tensor:
    """Return x with rotary position embedding applied along its sequence axis.

    Shapes: x (..., L, d), output (..., L, d) with the same dtype as x. Position t is the index
    along axis -2, counted from 0. Features form adjacent pairs (2i, 2i+1). Pair i of row t is
    rotated by the angle t * 10000**(-2i/d): read as complex numbers, it is multiplied by
    exp(i * angle).

    Parameters
    ----------
    x : torch.Tensor with at least 2 dimensions and a feature width d that is even and >= 2.
        Not modified.

    Returns
    -------
    torch.Tensor of the same shape and dtype as x.

    Raises
    ------
    ValueError
        If x has fewer than 2 dimensions, or if its last dimension is odd or zero.
    """
    raise NotImplementedError

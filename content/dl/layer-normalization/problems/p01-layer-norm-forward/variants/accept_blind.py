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
    if x.ndim == 0:
        raise ValueError("x must have at least one dimension")
    if not x.is_floating_point():
        raise ValueError("x must be a floating-point tensor")
    d = x.shape[-1]
    if d == 0:
        raise ValueError("the last dimension must not be empty")
    if tuple(weight.shape) != (d,) or tuple(bias.shape) != (d,):
        raise ValueError("weight and bias must have shape (D,)")
    if not eps > 0:
        raise ValueError("eps must be positive")

    mean = x.mean(dim=-1, keepdim=True)
    centred = x - mean
    var = (centred * centred).mean(dim=-1, keepdim=True)
    normed = centred / torch.sqrt(var + eps)
    return normed * weight.to(x.dtype) + bias.to(x.dtype)

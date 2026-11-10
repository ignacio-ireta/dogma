from __future__ import annotations

import torch


def _validate(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float) -> None:
    """Reject inputs outside the contract before any arithmetic runs."""
    if x.ndim < 1 or not x.is_floating_point():
        raise ValueError("x must be a floating-point tensor with at least one dimension")
    width = x.shape[-1]
    if width == 0:
        raise ValueError("the normalized (last) dimension must have at least one feature")
    if weight.shape != (width,) or bias.shape != (width,):
        raise ValueError("weight and bias must have shape (D,) matching the last dimension")
    if eps <= 0.0:
        raise ValueError("eps must be positive")


def layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Wrong on purpose: always returns float64, so a float32 input gets a float64 result."""
    _validate(x, weight, bias, eps)
    # Step 1: mean of each length-D vector, keeping the dimension so it broadcasts back.
    mean = x.mean(dim=-1, keepdim=True)
    # Step 2: centre first. Squaring after centring keeps the variance accurate under a large
    # common offset, where E[x^2] - mean^2 would cancel most of its digits.
    centred = x - mean
    # Step 3: biased variance over D, then the normalization with eps inside the square root.
    var = (centred * centred).mean(dim=-1, keepdim=True)
    normalized = centred / torch.sqrt(var + eps)
    # Step 4: per-feature affine transform; weight and bias broadcast over the leading dimensions.
    return (normalized * weight + bias).double()

from __future__ import annotations

import math

import torch


def ste_quantize(x: torch.Tensor, step: float) -> torch.Tensor:
    """Wrong on purpose: quantize x to multiples of step in the forward pass; pass gradients straight through.

    Forward: y = torch.round(x / step) * step (round half to even), same shape and dtype as x.
    Backward: dy/dx is the identity, so the incoming gradient reaches x unchanged (no clipping).
    """  # noqa: E501
    if isinstance(step, bool) or not isinstance(step, (int, float)):
        raise ValueError("step must be a number")
    if not math.isfinite(step) or step <= 0:
        raise ValueError("step must be finite and greater than zero")
    # Forward value: the staircase Q(x) = round(x / step) * step. round has zero slope almost
    # everywhere, so using it directly would block every gradient.
    quantized = torch.round(x / step) * step
    # Straight-through: the value is quantized, but the graph is that of the identity.
    # quantized - x is detached, so autograd sees d(output)/dx = 1 + 0 = 1.
    # Forward: x + (Q(x) - x) = Q(x). Backward: the gradient passes straight to x.
    return x + (quantized - x).detach()

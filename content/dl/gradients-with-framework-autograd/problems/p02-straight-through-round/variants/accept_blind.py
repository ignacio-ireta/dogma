from __future__ import annotations

import math

import torch


def ste_quantize(x: torch.Tensor, step: float) -> torch.Tensor:
    """Quantize x to multiples of step in the forward pass; pass gradients straight through.

    Forward: y = torch.round(x / step) * step (round half to even), same shape and dtype as x.
    Backward: dy/dx is the identity, so the incoming gradient reaches x unchanged (no clipping).
    y requires grad only if x does. Must not mutate x.
    Raises ValueError for non-floating x or for a step that is not finite and greater than zero.
    No torch.autograd.Function and no custom backward.
    """
    if not isinstance(x, torch.Tensor) or not x.is_floating_point():
        raise ValueError("x must be a floating tensor")
    if (
        isinstance(step, bool)
        or not isinstance(step, (int, float))
        or not math.isfinite(step)
        or step <= 0
    ):
        raise ValueError("step must be a finite number greater than zero")

    quantized = torch.round(x / step) * step
    return x + (quantized - x).detach()

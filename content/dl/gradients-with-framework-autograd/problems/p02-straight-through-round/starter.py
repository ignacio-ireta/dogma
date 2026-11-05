from __future__ import annotations

import torch


def ste_quantize(x: torch.Tensor, step: float) -> torch.Tensor:
    """Quantize x to multiples of step in the forward pass; pass gradients straight through.

    Forward: y = torch.round(x / step) * step (round half to even), same shape and dtype as x.
    Backward: dy/dx is the identity, so the incoming gradient reaches x unchanged (no clipping).
    y requires grad only if x does. Must not mutate x.
    Raises ValueError for non-floating x or for a step that is not finite and greater than zero.
    No torch.autograd.Function and no custom backward.
    """
    raise NotImplementedError

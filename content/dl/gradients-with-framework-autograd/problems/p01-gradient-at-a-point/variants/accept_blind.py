from __future__ import annotations

from collections.abc import Callable

import torch


def grad_at(f: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    """Gradient of a scalar function f at the 1-D point x, computed with torch.autograd.grad.

    f maps a 1-D float tensor of shape (n,) to a 0-dimensional tensor. Returns a detached tensor of
    shape (n,) with the dtype of x, equal to df/dx at x. If f does not depend on x, returns zeros.
    Leaves x unchanged: its values, requires_grad flag and .grad field. Works inside no_grad.
    Raises ValueError for non-1-D or non-floating x, or for a non-scalar f(x).
    """
    if not isinstance(x, torch.Tensor) or x.dim() != 1 or not x.is_floating_point():
        raise ValueError("x must be a 1-D floating tensor")

    with torch.enable_grad():
        point = x.detach().clone().requires_grad_(True)
        out = f(point)
        if not isinstance(out, torch.Tensor) or out.dim() != 0:
            raise ValueError("f(x) must be a 0-dimensional tensor")
        if not out.requires_grad:
            return torch.zeros_like(x).detach()
        (grad,) = torch.autograd.grad(out, point, allow_unused=True)
        if grad is None:
            return torch.zeros_like(x).detach()
    return grad.detach().to(dtype=x.dtype)

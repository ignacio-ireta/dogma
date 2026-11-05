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
    raise NotImplementedError

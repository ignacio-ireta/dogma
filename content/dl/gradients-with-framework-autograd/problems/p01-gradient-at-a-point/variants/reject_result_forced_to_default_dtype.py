from __future__ import annotations

from collections.abc import Callable

import torch


def grad_at(f: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    """Wrong on purpose: gradient of a scalar function f at the 1-D point x, computed with torch.autograd.grad.

    f maps a 1-D float tensor of shape (n,) to a 0-dimensional tensor. Returns a detached tensor of
    shape (n,) with the dtype of x, equal to df/dx at x. If f does not depend on x, returns zeros.
    """  # noqa: E501
    if not isinstance(x, torch.Tensor) or x.ndim != 1 or not x.is_floating_point():
        raise ValueError("x must be a 1-D floating tensor")
    # The caller may be inside no_grad(). Gradients need a recorded graph, so turn recording on.
    with torch.enable_grad():
        # A private leaf copy: the graph starts here, so the caller's tensor is never touched.
        # Its requires_grad flag, values and .grad field stay as they were.
        leaf = x.detach().clone().requires_grad_(True)
        out = f(leaf)
        if out.ndim != 0:
            raise ValueError("f must return a 0-dimensional (scalar) tensor")
        # If the output does not depend on the leaf, there is no graph to differentiate.
        # The derivative of a constant is zero, so return zeros instead of raising.
        if not out.requires_grad:
            return torch.zeros_like(x)
        # torch.autograd.grad returns the gradient as a new tensor. It does not write to leaf.grad,
        # which is why no state leaks back to the caller (unlike Tensor.backward).
        (grad,) = torch.autograd.grad(out, leaf, allow_unused=True)
    if grad is None:  # f built a graph but never used x: the gradient is zero, not missing
        return torch.zeros_like(x)
    # The returned gradient is already detached from any graph, and it is not created with
    # create_graph=True, so it carries no grad_fn.
    return grad.detach().type(torch.get_default_dtype())

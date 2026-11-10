from __future__ import annotations

import torch


def _validate(  # noqa: PLR0917 - mirrors the contract arguments
    x: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
    w_shortcut: torch.Tensor | None,
) -> None:
    """Reject bad input before any arithmetic, so errors name the broken rule."""
    tensors = [x, w1, b1, w2, b2] + ([w_shortcut] if w_shortcut is not None else [])
    if not all(isinstance(t, torch.Tensor) for t in tensors):
        raise ValueError("all arguments must be torch tensors")
    if not x.is_floating_point() or x.ndim < 1:
        raise ValueError("x must be a floating-point tensor with a feature axis")
    if any(t.dtype != x.dtype for t in tensors):
        raise ValueError("all tensors must share the dtype of x")
    d_in = x.shape[-1]
    if d_in < 1:
        raise ValueError("feature dimension must be at least 1")
    if w1.ndim != 2 or w1.shape[0] != d_in:
        raise ValueError("w1 must have shape (d_in, h)")
    h = w1.shape[1]
    if b1.shape != (h,):
        raise ValueError("b1 must have shape (h,) matching w1")
    if w2.ndim != 2 or w2.shape[0] != h or w2.shape[1] < 1:
        raise ValueError("w2 must have shape (h, d_out)")
    d_out = w2.shape[1]
    if b2.shape != (d_out,):
        raise ValueError("b2 must have shape (d_out,) matching w2")
    if w_shortcut is None:
        if d_out != d_in:
            raise ValueError(
                "identity shortcut needs d_out == d_in; pass w_shortcut for a projection"
            )
    elif w_shortcut.shape != (d_in, d_out):
        raise ValueError("w_shortcut must have shape (d_in, d_out)")


def residual_block(  # noqa: PLR0917 - the contract fixes six explicit arguments
    x: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
    w_shortcut: torch.Tensor | None = None,
) -> torch.Tensor:
    """Wrong on purpose: accepts a zero-width hidden layer instead of raising ValueError."""
    _validate(x, w1, b1, w2, b2, w_shortcut)
    # Residual branch F(x): dense -> ReLU -> dense. Lesson: "The system", F is the learned part.
    hidden = torch.relu(x @ w1 + b1)
    branch = hidden @ w2 + b2
    # Shortcut S(x): the identity when widths agree, otherwise a learned projection.
    # Using x directly (no detach) keeps the identity term I in dy/dx = I + dF/dx.
    shortcut = x if w_shortcut is None else x @ w_shortcut
    # The sum creates a new tensor, so the inputs are never written to.
    return shortcut + branch

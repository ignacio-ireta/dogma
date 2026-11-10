from __future__ import annotations

import torch


def residual_block(  # noqa: PLR0912, PLR0917
    x: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
    w_shortcut: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return shortcut(x) + F(x) with F(x) = relu(x @ w1 + b1) @ w2 + b2.

    x: floating (..., d_in); w1: (d_in, h); b1: (h,); w2: (h, d_out); b2: (d_out,).
    w_shortcut: None (identity, needs d_out == d_in) or (d_in, d_out) projection.
    Every tensor shares x's floating dtype. Raise ValueError on invalid input.
    Never mutate inputs and never detach them: the identity term must reach x's gradient.
    """
    if not isinstance(x, torch.Tensor) or not x.is_floating_point():
        raise ValueError("x must be a floating-point tensor")
    if x.dim() < 1:
        raise ValueError("x must have at least one dimension")
    d_in = x.shape[-1]
    dtype = x.dtype

    for name, t in (("w1", w1), ("b1", b1), ("w2", w2), ("b2", b2)):
        if not isinstance(t, torch.Tensor) or t.dtype != dtype:
            raise ValueError(f"{name} must be a tensor with the dtype of x")

    if w1.dim() != 2 or w1.shape[0] != d_in:
        raise ValueError("w1 must have shape (d_in, h)")
    h = w1.shape[1]
    if b1.shape != (h,):
        raise ValueError("b1 must have shape (h,)")
    if w2.dim() != 2 or w2.shape[0] != h:
        raise ValueError("w2 must have shape (h, d_out)")
    d_out = w2.shape[1]
    if b2.shape != (d_out,):
        raise ValueError("b2 must have shape (d_out,)")

    if d_in == 0 or h == 0 or d_out == 0:
        raise ValueError("d_in, h and d_out must be non-zero")

    if w_shortcut is None:
        if d_out != d_in:
            raise ValueError(
                "identity shortcut needs d_out == d_in; pass w_shortcut for a projection"
            )
        shortcut = x
    else:
        if not isinstance(w_shortcut, torch.Tensor) or w_shortcut.dtype != dtype:
            raise ValueError("w_shortcut must be a tensor with the dtype of x")
        if w_shortcut.shape != (d_in, d_out):
            raise ValueError("w_shortcut must have shape (d_in, d_out)")
        shortcut = x @ w_shortcut

    hidden = torch.relu(x @ w1 + b1)
    branch = hidden @ w2 + b2
    return shortcut + branch

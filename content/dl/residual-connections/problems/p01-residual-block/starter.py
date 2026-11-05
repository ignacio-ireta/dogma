from __future__ import annotations

import torch


def residual_block(  # noqa: PLR0917 - the contract fixes six explicit arguments
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
    raise NotImplementedError

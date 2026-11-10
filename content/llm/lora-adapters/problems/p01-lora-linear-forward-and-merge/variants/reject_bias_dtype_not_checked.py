from __future__ import annotations

import math

import torch


def _check_real(alpha: object) -> float:
    # bool is a subclass of int, so it is rejected by name. A NaN or infinite scale would turn
    # every output into NaN or inf without a visible error.
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise ValueError("alpha must be a real number")
    if not math.isfinite(alpha):
        raise ValueError("alpha must be finite")
    return float(alpha)


def lora_linear(  # noqa: PLR0917 (the contract fixes seven arguments)
    x: torch.Tensor,
    W: torch.Tensor,
    b: torch.Tensor | None,
    A: torch.Tensor,
    B: torch.Tensor,
    alpha: float,
    merge: bool = False,
) -> torch.Tensor:
    """Wrong on purpose: LoRA linear forward (see problem.md): y = x W^T + b + (alpha / r) x A^T B^T."""  # noqa: E501
    # Validate the interface first, so a silent broadcast cannot hide a wrong shape.
    if x.ndim < 1:
        raise ValueError("x must have at least one dimension (..., d_in)")
    if W.ndim != 2:
        raise ValueError("W must be 2-D (d_out, d_in)")
    d_out, d_in = W.shape
    if x.shape[-1] != d_in:
        raise ValueError("x.shape[-1] must equal W.shape[1] (d_in)")
    if A.ndim != 2 or A.shape[1] != d_in:
        raise ValueError("A must have shape (r, d_in)")
    r = A.shape[0]
    if r < 1:
        raise ValueError("the adapter rank r must be at least 1")
    if B.ndim != 2 or tuple(B.shape) != (d_out, r):
        raise ValueError(f"B must have shape (d_out, r) = ({d_out}, {r})")
    if b is not None and tuple(b.shape) != (d_out,):
        raise ValueError("b must have shape (d_out,) or be None")
    scale = _check_real(alpha) / r  # alpha / r, with r read from A, not from the caller
    tensors = [x, W, A, B]
    if not all(t.is_floating_point() for t in tensors):
        raise ValueError("x, W, A, B and b must be floating-point tensors")
    if len({t.dtype for t in tensors}) != 1:
        raise ValueError("x, W, A, B and b must share one dtype")

    if merge:
        # Fold the update into one weight: W' = W + (alpha / r) B A, shape (d_out, r) @ (r, d_in).
        # The product is built as a new tensor, so W is left untouched.
        W_eff = W + scale * (B @ A)
        y = x @ W_eff.T
    else:
        # Two thin products. x A^T has shape (..., r); times B^T gives (..., d_out). The full
        # (d_out, d_in) update is never formed on this path.
        y = x @ W.T + scale * ((x @ A.T) @ B.T)

    # The bias is added once, after either path, and is not scaled by alpha / r.
    if b is not None:
        y = y + b
    return y

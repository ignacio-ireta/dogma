from __future__ import annotations

import math

import torch


def gelu_ffn(
    X: torch.Tensor,
    W1: torch.Tensor,
    b1: torch.Tensor,
    W2: torch.Tensor,
    b2: torch.Tensor,
) -> torch.Tensor:
    """Position-wise FFN: W2 @ GELU(W1 @ x + b1) + b2, applied to every position.

    Shapes: X (..., d_model), W1 (d_ff, d_model), b1 (d_ff,), W2 (d_model, d_ff), b2 (d_model,).
    """
    # Validate the interface. Positions are the leading axes; the feature axis is the last one.
    if X.ndim < 1:
        raise ValueError("X must have at least one dimension (..., d_model)")
    if W1.ndim != 2:
        raise ValueError("W1 must be 2-D (d_ff, d_model)")
    d_ff, d_model = W1.shape
    if X.shape[-1] != d_model:
        raise ValueError("the last dimension of X must equal W1.shape[1] (d_model)")
    if tuple(W2.shape) != (d_model, d_ff):
        raise ValueError("W2 must have shape (d_model, d_ff)")
    if tuple(b1.shape) != (d_ff,):
        raise ValueError("b1 must have shape (d_ff,)")
    if tuple(b2.shape) != (d_model,):
        raise ValueError("b2 must have shape (d_model,)")

    # Expansion: z = x @ W1.T + b1, shape (..., d_ff). nn.Linear stores W1 as (out, in), so the
    # transpose maps d_model features to d_ff features.
    z = X @ W1.T + b1

    # Exact GELU: GELU(z) = z * Phi(z) = 0.5 * z * (1 + erf(z / sqrt(2))). Phi is the standard
    # normal CDF, written with the error function. This is not the tanh approximation.
    gelu = 0.5 * z * (1.0 + torch.erf(z / math.sqrt(2.0)))

    # Projection back: y = h @ W2.T + b2, shape (..., d_model). The matrix product touches only
    # the feature axis, so every position is handled independently.
    return gelu @ W2.T + b2

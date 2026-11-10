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
    """Apply the position-wise FFN  W2 @ GELU(W1 @ x + b1) + b2  to every position of X."""
    if X.dim() == 0:
        raise ValueError("X must have at least one dimension")
    if W1.dim() != 2:
        raise ValueError("W1 must be 2-D (d_ff, d_model)")
    d_ff, d_model = W1.shape
    if X.shape[-1] != d_model:
        raise ValueError("last dimension of X must equal W1.shape[1]")
    if tuple(b1.shape) != (d_ff,):
        raise ValueError("b1 must have shape (d_ff,)")
    if tuple(W2.shape) != (d_model, d_ff):
        raise ValueError("W2 must have shape (d_model, d_ff)")
    if tuple(b2.shape) != (d_model,):
        raise ValueError("b2 must have shape (d_model,)")

    z = X @ W1.T + b1
    # Exact GELU: z * Phi(z), with Phi(z) = 0.5 * erfc(-z / sqrt(2)), which equals
    # 0.5 * (1 + erf(z / sqrt(2))) without cancellation in the negative tail.
    phi = 0.5 * torch.erfc(-z / math.sqrt(2.0))
    h = z * phi
    return h @ W2.T + b2

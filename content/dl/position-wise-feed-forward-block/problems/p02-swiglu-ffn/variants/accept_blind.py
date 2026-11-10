from __future__ import annotations

import torch


def swiglu_ffn(
    X: torch.Tensor,
    W: torch.Tensor,
    V: torch.Tensor,
    W2: torch.Tensor,
) -> torch.Tensor:
    """Apply the gated feed-forward block  W2 @ (SiLU(W @ x) * (V @ x))  to every position."""
    if X.dim() == 0:
        raise ValueError("X must have at least one dimension")
    if W.dim() != 2 or V.dim() != 2:
        raise ValueError("W and V must be 2-D")
    if W.shape != V.shape:
        raise ValueError("W and V must have the same shape")
    d_ff, d_model = W.shape
    if X.shape[-1] != d_model:
        raise ValueError("last dimension of X must equal W.shape[1]")
    if tuple(W2.shape) != (d_model, d_ff):
        raise ValueError("W2 must have shape (d_model, d_ff)")

    gate = X @ W.T
    up = X @ V.T
    silu = gate * torch.sigmoid(gate)
    return (silu * up) @ W2.T

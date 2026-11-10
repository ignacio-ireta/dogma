from __future__ import annotations

import torch


def swiglu_ffn(
    X: torch.Tensor,
    W: torch.Tensor,
    V: torch.Tensor,
    W2: torch.Tensor,
) -> torch.Tensor:
    """Gated SwiGLU block: W2 @ (SiLU(W x) * (V x)), per position, with no biases.

    Shapes: X (..., d_model), W and V (d_ff, d_model), W2 (d_model, d_ff), output (..., d_model).
    """
    # Validate the interface. W and V must match because they are multiplied elementwise.
    if X.ndim < 1:
        raise ValueError("X must have at least one dimension (..., d_model)")
    if W.ndim != 2 or V.ndim != 2:
        raise ValueError("W and V must be 2-D (d_ff, d_model)")
    if tuple(W.shape) != tuple(V.shape):
        raise ValueError("W and V must have the same shape (d_ff, d_model)")
    d_ff, d_model = W.shape
    if X.shape[-1] != d_model:
        raise ValueError("the last dimension of X must equal W.shape[1] (d_model)")
    if tuple(W2.shape) != (d_model, d_ff):
        raise ValueError("W2 must have shape (d_model, d_ff)")

    # Two linear maps read the same input. nn.Linear stores (out, in), hence the transposes.
    gate = torch.einsum(
        "...i,oi->...o", X, W
    )  # (..., d_ff): this path is passed through the nonlinearity
    up = torch.einsum("...i,oi->...o", X, V)  # (..., d_ff): this path stays linear

    # SiLU(z) = z * sigmoid(z) (also called Swish with beta = 1). It is applied to the gate only.
    # torch.sigmoid is numerically stable on both tails.
    silu = gate * torch.sigmoid(gate)

    # Gate the up path with the activated gate, elementwise. Shape (..., d_ff).
    hidden = silu * up

    # Project back to d_model. No bias terms in this block.
    return torch.einsum("...h,dh->...d", hidden, W2)

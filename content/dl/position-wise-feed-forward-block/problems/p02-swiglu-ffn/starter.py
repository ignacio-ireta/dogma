from __future__ import annotations

import torch


def swiglu_ffn(
    X: torch.Tensor,
    W: torch.Tensor,
    V: torch.Tensor,
    W2: torch.Tensor,
) -> torch.Tensor:
    """Apply the gated feed-forward block  W2 @ (SiLU(W @ x) * (V @ x))  to every position.

    Shapes: X (..., d_model); W and V (d_ff, d_model) in nn.Linear layout; W2 (d_model, d_ff).
    SiLU(z) = z * sigmoid(z). The block has no biases. Output has shape (..., d_model).

    Parameters
    ----------
    X : float64 tensor with at least one dimension. Leading axes are independent positions.
    W, V : float64 tensors (d_ff, d_model). W is the gate projection, V the up projection.
    W2 : float64 tensor (d_model, d_ff). d_ff may be 0.

    Returns
    -------
    float64 tensor of shape X.shape. Inputs are not modified.

    Raises
    ------
    ValueError
        If X is 0-dimensional, W or V is not 2-D or they differ in shape, the last dimension of X
        is not d_model, or W2 is not (d_model, d_ff).
    """
    raise NotImplementedError

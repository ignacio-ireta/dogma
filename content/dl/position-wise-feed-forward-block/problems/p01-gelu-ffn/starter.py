from __future__ import annotations

import torch


def gelu_ffn(
    X: torch.Tensor,
    W1: torch.Tensor,
    b1: torch.Tensor,
    W2: torch.Tensor,
    b2: torch.Tensor,
) -> torch.Tensor:
    """Apply the position-wise FFN  W2 @ GELU(W1 @ x + b1) + b2  to every position of X.

    Shapes: X (..., d_model); W1 (d_ff, d_model) and b1 (d_ff,) in nn.Linear layout;
    W2 (d_model, d_ff) and b2 (d_model,). Output has the same shape as X.
    GELU is the exact form z * Phi(z) with Phi(z) = 0.5 * (1 + erf(z / sqrt(2))).

    Parameters
    ----------
    X : float64 tensor with at least one dimension. Leading axes are independent positions.
    W1, b1, W2, b2 : float64 tensors as above. d_ff may be 0.

    Returns
    -------
    float64 tensor of shape X.shape. Inputs are not modified.

    Raises
    ------
    ValueError
        If X is 0-dimensional or its last dimension is not d_model, if W1 is not 2-D, or if any
        of W2, b1 or b2 does not have the shape the contract gives.
    """
    raise NotImplementedError

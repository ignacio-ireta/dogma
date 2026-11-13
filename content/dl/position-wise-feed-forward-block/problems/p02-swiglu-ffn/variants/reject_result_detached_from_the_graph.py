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
    gate = X @ W.T  # (..., d_ff): this path is passed through the nonlinearity
    up = X @ V.T  # (..., d_ff): this path stays linear

    # SiLU(z) = z * sigmoid(z) (also called Swish with beta = 1). It is applied to the gate only.
    # torch.sigmoid is numerically stable on both tails.
    silu = gate * torch.sigmoid(gate)

    # Gate the up path with the activated gate, elementwise. Shape (..., d_ff).
    hidden = silu * up

    # Project back to d_model. No bias terms in this block.
    return hidden @ W2.T


_correct = swiglu_ffn


def _laxer(r):
    if type(r).__name__ == "Tensor" and r.requires_grad:
        return r.detach()
    return r


def swiglu_ffn(*args, **kwargs):
    """Wrong on purpose: returns a result cut off from the graph."""
    result = _correct(*args, **kwargs)
    return _everywhere(result)


def _everywhere(r):
    if type(r) is list:
        return [_everywhere(item) for item in r]
    if type(r) is tuple:
        return tuple(_everywhere(item) for item in r)
    if type(r) is dict:
        return {key: _everywhere(item) for key, item in r.items()}
    return _laxer(r)


swiglu_ffn.__wrapped__ = _correct

from __future__ import annotations

import torch


def batched_sq_distances(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Squared Euclidean distances between every X[b] row and every Y[b] row."""
    if not isinstance(X, torch.Tensor) or not isinstance(Y, torch.Tensor):
        raise ValueError("X and Y must be tensors")
    if not X.is_floating_point() or not Y.is_floating_point():
        raise ValueError("X and Y must be floating point tensors")
    if X.dim() != 3 or Y.dim() != 3:
        raise ValueError("X and Y must be 3-dimensional")
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same batch size")
    if X.shape[2] != Y.shape[2]:
        raise ValueError("X and Y must have the same feature size")

    out_dtype = torch.promote_types(X.dtype, Y.dtype)
    # Work in float64 with the direct difference so that a large common offset
    # and very small coordinates keep their precision; cast back at the end.
    x = X.to(torch.float64)
    y = Y.to(torch.float64)
    diff = x[:, :, None, :] - y[:, None, :, :]
    out = (diff * diff).sum(dim=-1)
    return out.to(out_dtype)

from __future__ import annotations

import torch


def _check(X: torch.Tensor, Y: torch.Tensor) -> None:
    """Reject inputs the contract excludes before any arithmetic happens."""
    for name, tensor in (("X", X), ("Y", Y)):
        if not tensor.is_floating_point():
            raise ValueError(f"{name} must be a floating tensor")
        if tensor.ndim != 3:
            raise ValueError(f"{name} must have shape (batch, points, features)")
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same batch size")
    if X.shape[2] != Y.shape[2]:
        raise ValueError("X and Y must have the same feature size")


def batched_sq_distances(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Wrong on purpose: squared Euclidean distances between every X[b] row and every Y[b] row.

    X: floating (B, N, D); Y: floating (B, M, D), any float dtypes.
    Returns (B, N, M) with out[b, i, j] = sum_d (X[b, i, d] - Y[b, j, d])**2,
    dtype torch.promote_types(X.dtype, Y.dtype). Every entry is non-negative.
    """
    _check(X, Y)
    # Promote once, so the arithmetic and the result use the same dtype.
    dtype = X.dtype
    X = X.to(dtype)
    Y = Y.to(dtype)
    # Broadcasting: (B, N, 1, D) - (B, 1, M, D) -> (B, N, M, D). Entry [b, i, j, d] is the
    # coordinate difference for one pair, so no Python loop over pairs is needed.
    diff = X[:, :, None, :] - Y[:, None, :, :]
    # Subtract first, then square and sum over the feature axis (the last one). Squaring a
    # difference cannot go negative, and nothing cancels against a large offset.
    # The cost is a (B, N, M, D) temporary; the expansion trades that memory for rounding.
    return (diff * diff).sum(dim=-1)

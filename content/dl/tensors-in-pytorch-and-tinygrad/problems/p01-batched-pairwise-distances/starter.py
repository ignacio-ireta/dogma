from __future__ import annotations

import torch


def batched_sq_distances(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Squared Euclidean distances between every X[b] row and every Y[b] row.

    X: floating (B, N, D); Y: floating (B, M, D), any float dtypes.
    Returns (B, N, M) with out[b, i, j] = sum_d (X[b, i, d] - Y[b, j, d])**2,
    dtype torch.promote_types(X.dtype, Y.dtype). Every entry is non-negative.
    Raises ValueError for non-floating input, non-3-D input, or mismatched B or D.
    Must not mutate X or Y. No Python loop over batches or pairs; no torch.cdist, no torch.nn.
    """
    raise NotImplementedError

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def pairwise_sq_distances(X: ArrayLike, Y: ArrayLike) -> NDArray[np.float64]:
    """Return the (n, m) matrix of squared Euclidean distances between rows of X and Y."""
    x = np.asarray(X, dtype=np.float64)
    y = np.asarray(Y, dtype=np.float64)

    # Check shapes before broadcasting. A size-1 feature axis would otherwise stretch
    # silently against the other side, so the feature counts must match explicitly.
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError(f"expected 2-D inputs, got shapes {x.shape} and {y.shape}")
    if x.shape[1] != y.shape[1]:
        raise ValueError(f"feature counts differ: X has {x.shape[1]} features, Y has {y.shape[1]}")

    # x[:, None, :] is (n, 1, d) and y[None, :, :] is (1, m, d). Broadcasting stretches
    # the length-1 axes, so the difference is (n, m, d): every pair, every coordinate.
    # Subtracting the points directly means no large norms are formed, so nothing cancels.
    diff = x[:, None, :] - y[None, :, :]

    # Sum the squared coordinates over the feature axis (the last one). Each squared
    # term is >= 0, so every entry of the result is >= 0 by construction.
    # The cost is O(n * m * d) memory; the expansion in the lesson avoids that.
    return np.einsum("ijk,ijk->ij", diff, diff)

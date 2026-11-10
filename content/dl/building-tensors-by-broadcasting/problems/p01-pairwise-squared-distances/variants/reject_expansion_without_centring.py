from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def pairwise_sq_distances(X: ArrayLike, Y: ArrayLike) -> NDArray[np.float64]:
    """Wrong on purpose: uses the norm expansion, which cancels badly when points sit far from the origin."""  # noqa: E501
    x = np.asarray(X, dtype=np.float64)
    y = np.asarray(Y, dtype=np.float64)

    # Check shapes before broadcasting. A size-1 feature axis would otherwise stretch
    # silently against the other side, so the feature counts must match explicitly.
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError(f"expected 2-D inputs, got shapes {x.shape} and {y.shape}")
    if x.shape[1] != y.shape[1]:
        raise ValueError(f"feature counts differ: X has {x.shape[1]} features, Y has {y.shape[1]}")

    # Expansion |x|^2 + |y|^2 - 2 x.y: large norms cancel, and the clamp hides negatives.
    sq = np.sum(x * x, axis=1)[:, None] + np.sum(y * y, axis=1)[None, :] - 2.0 * (x @ y.T)
    return np.maximum(sq, 0.0)

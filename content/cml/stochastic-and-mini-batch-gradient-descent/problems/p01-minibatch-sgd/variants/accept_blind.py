from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def minibatch_sgd(
    X: ArrayLike,
    y: ArrayLike,
    lr: float,
    batch_size: int,
    epochs: int,
    *,
    seed: int,
) -> NDArray[np.float64]:
    """Fit least squares without intercept by mini-batch gradient descent."""
    A = np.array(X, dtype=np.float64)
    t = np.array(y, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError("X must be 2-D")
    if t.ndim != 1 or t.shape[0] != A.shape[0]:
        raise ValueError("y must be 1-D with one entry per row of X")
    n, d = A.shape
    if not (1 <= batch_size <= n):
        raise ValueError("batch_size must be between 1 and n")
    if not lr > 0:
        raise ValueError("lr must be positive")
    if epochs < 0:
        raise ValueError("epochs must be non-negative")

    w = np.zeros(d, dtype=np.float64)
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        perm = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            Xb = A[idx]
            yb = t[idx]
            g = Xb.T @ (Xb @ w - yb) / len(idx)
            w = w - lr * g
    return w

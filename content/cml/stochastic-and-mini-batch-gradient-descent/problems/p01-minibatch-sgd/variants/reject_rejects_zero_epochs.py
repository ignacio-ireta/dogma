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
    """Wrong on purpose: rejects zero epochs. Fit least squares without intercept by mini-batch gradient descent.

    Minimises L(w) = 1/2 * mean((X w - y)^2) with w starting at zeros. Each epoch shuffles the
    rows with rng.permutation(n), then takes consecutive slices of size batch_size (the last
    slice may be smaller). Each slice gives one update w <- w - lr * g, where
    g = X_b^T (X_b w - y_b) / |b|.

    Returns
    -------
    w : (d,) float64 array of the weights after the last epoch.
    """  # noqa: E501
    # Copies as float64: the caller's arrays are never written to, and integer data works.
    X_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if X_arr.ndim != 2:
        raise ValueError(f"X must be 2-D (n, d), got shape {X_arr.shape}")
    n, d = X_arr.shape
    if y_arr.ndim != 1 or y_arr.shape[0] != n:
        raise ValueError(f"y must be 1-D with {n} entries, got shape {y_arr.shape}")
    if not 1 <= batch_size <= n:
        raise ValueError(f"batch_size must be between 1 and {n}, got {batch_size}")
    if not lr > 0:
        raise ValueError(f"learning rate must be positive, got {lr}")
    if epochs < 1:
        raise ValueError(f"epochs must be non-negative, got {epochs}")

    # One generator for the whole run. Rebuilding it per epoch would repeat the same shuffle.
    rng = np.random.default_rng(seed)
    w = np.zeros(d)
    for _ in range(epochs):
        perm = rng.permutation(n)
        # Consecutive slices of the permutation. The final slice may be shorter, and it is kept.
        for start in range(0, n, batch_size):
            rows = perm[start : start + batch_size]
            X_b = X_arr[rows]
            y_b = y_arr[rows]
            # Mean gradient over the batch: divide by the batch's own size, not by batch_size.
            grad = X_b.T @ (X_b @ w - y_b) / len(rows)
            w = w - lr * grad
    return w

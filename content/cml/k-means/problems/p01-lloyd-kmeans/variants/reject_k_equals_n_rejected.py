"""Wrong on purpose: Lloyd's algorithm from given centroids (lesson: The system)."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _finite_matrix(value: ArrayLike, name: str) -> NDArray[np.float64]:
    """A finite 2-D float array with at least one row and one column."""
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"{name} must be a 2-D array with at least one row and one column")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _assign(x: NDArray[np.float64], centroids: NDArray[np.float64]) -> NDArray[np.int64]:
    """Step 1: nearest centroid by squared distance; argmin keeps the lower index on ties."""
    d2 = np.sum((x[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
    return np.argmin(d2, axis=1)


def _update(
    x: NDArray[np.float64], labels: NDArray[np.int64], centroids: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Step 2: each centroid becomes its samples' mean; an empty cluster keeps its centroid."""
    counts = np.bincount(labels, minlength=len(centroids))
    sums = np.zeros_like(centroids)
    np.add.at(sums, labels, x)
    new = centroids.copy()
    filled = counts > 0
    new[filled] = sums[filled] / counts[filled, None]
    return new


def kmeans(
    X: ArrayLike,
    init_centroids: ArrayLike,
    *,
    max_iter: int = 300,
    tol: float = 1e-6,
) -> tuple[NDArray[np.float64], NDArray[np.int64], float, int]:
    """Run Lloyd's algorithm from given centroids and return the four results."""
    x = _finite_matrix(X, "X")
    c0 = _finite_matrix(init_centroids, "init_centroids")
    if c0.shape[1] != x.shape[1]:
        raise ValueError("init_centroids must have the same number of columns as X")
    if not 1 <= c0.shape[0] < x.shape[0]:
        raise ValueError("init_centroids must have between 1 and n rows")
    if isinstance(max_iter, bool) or not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError("max_iter must be an integer of at least 1")
    tol_value = float(tol)
    if not math.isfinite(tol_value) or tol_value < 0.0:
        raise ValueError("tol must be a finite number that is zero or positive")

    centroids = c0.copy()
    n_iter = 0
    for iteration in range(1, max_iter + 1):
        labels = _assign(x, centroids)  # step 1
        new = _update(x, labels, centroids)  # step 2
        # Step 3: the largest Euclidean move of any centroid in this iteration.
        shift = float(np.max(np.linalg.norm(new - centroids, axis=1)))
        centroids = new
        n_iter = iteration
        if shift <= tol_value:
            break

    # Labels and inertia are reported against the returned centroids.
    labels = _assign(x, centroids)
    inertia = float(np.sum((x - centroids[labels]) ** 2))
    return centroids, labels.astype(np.int64), inertia, n_iter

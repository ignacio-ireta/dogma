"""Wrong on purpose: k-means++ seeding with the exact protocol (lesson: The system)."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def kmeans_plus_plus(X: ArrayLike, k: int, *, seed: int) -> NDArray[np.float64]:
    """Return k centres chosen by k-means++ with rng = np.random.default_rng(seed)."""
    x = np.asarray(X, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0:
        raise TypeError("X must be a 2-D array with at least one row and one column")
    if not np.all(np.isfinite(x)):
        raise ValueError("X must contain only finite values")
    n = x.shape[0]
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or not 1 <= k <= n:
        raise ValueError("k must be an integer with 1 <= k <= n")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if np.unique(x, axis=0).shape[0] < k:
        raise ValueError("X must have at least k distinct rows")

    # A common power-of-two scale preserves all D^2 probabilities while preventing the
    # squaring step from overflowing or underflowing solely because of the data's units.
    exponent = int(np.frexp(np.max(np.abs(x)))[1])
    distance_x = np.ldexp(x, -exponent)
    rng = np.random.default_rng(int(seed))
    # Step 2: the first centre is a uniformly drawn row.
    first = int(rng.integers(n))
    chosen = [first]
    # d2[i] is the squared distance from row i to its nearest centre chosen so far.
    d2 = np.sum((distance_x - distance_x[first]) ** 2, axis=1)
    for _ in range(1, k):
        # Step 3: sample a row with probability proportional to D^2. Chosen rows have D^2 = 0.
        idx = int(rng.choice(n, p=d2 / d2.sum()))
        chosen.append(idx)
        # Keep the minimum over all chosen centres, so D^2 always refers to the nearest one.
        d2 = np.minimum(d2, np.sum((distance_x - distance_x[idx]) ** 2, axis=1))
    return x[chosen].copy()

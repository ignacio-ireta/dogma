from __future__ import annotations

import numbers

import numpy as np


def kmeans_plus_plus(X, k, *, seed):
    A = np.asarray(X)
    if A.ndim != 2 or A.shape[0] < 1 or A.shape[1] < 1:
        raise ValueError("X must be 2-D and non-empty")
    try:
        A = A.astype(np.float64)
    except (TypeError, ValueError) as e:
        raise ValueError("X must be numeric") from e
    if not np.all(np.isfinite(A)):
        raise ValueError("X must be finite")
    n = A.shape[0]
    if isinstance(k, bool) or not isinstance(k, numbers.Integral) or not 1 <= k <= n:
        raise ValueError("k must be between 1 and the number of rows")
    if isinstance(seed, bool) or not isinstance(seed, numbers.Integral) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    k = int(k)
    if np.unique(A, axis=0).shape[0] < k:
        raise ValueError("X must have at least k distinct rows")
    mx = float(np.max(np.abs(A)))
    if mx > 0:
        e = int(np.frexp(mx)[1])
        S = np.ldexp(A, -e)
    else:
        S = A
    rng = np.random.default_rng(int(seed))
    idxs = [int(rng.integers(n))]
    d2 = ((S - S[idxs[0]]) ** 2).sum(axis=1)
    for _ in range(1, k):
        p = d2 / d2.sum()
        idx = int(rng.choice(n, p=p))
        idxs.append(idx)
        d2 = np.minimum(d2, ((S - S[idx]) ** 2).sum(axis=1))
    return A[idxs].copy()

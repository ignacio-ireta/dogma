import numpy as np


def covariance_matrix(X, ddof=1):
    A = np.asarray(X, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError("X must be 2-D")
    n, d = A.shape
    if n == 0 or d == 0:
        raise ValueError("X must have rows and columns")
    if not np.isfinite(A).all():
        raise ValueError("X must be finite")
    if isinstance(ddof, bool) or not isinstance(ddof, (int, np.integer)) or ddof < 0:
        raise ValueError("ddof must be a non-negative integer")
    if n <= ddof:
        raise ValueError("need more observations than ddof")
    mu = A.mean(axis=0)
    out = np.zeros((d, d))
    for i in range(n):
        dev = A[i] - mu
        out += np.outer(dev, dev)
    return out / (n - ddof)

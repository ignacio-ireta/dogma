import numpy as np


def correlation_matrix(X):
    A = np.asarray(X, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError("X must be 2-D")
    n, d = A.shape
    if n == 0 or d == 0:
        raise ValueError("X must have rows and columns")
    if n < 2:
        raise ValueError("need at least two observations")
    if not np.isfinite(A).all():
        raise ValueError("X must be finite")
    peak = np.abs(A).max(axis=0)
    peak = np.where(peak > 0, peak, 1.0)
    A = A / peak  # correlation is scale-free; avoids overflow in the squares
    R = np.empty((d, d))
    for j in range(d):
        for k in range(d):
            if np.all(A[:, j] == A[0, j]) or np.all(A[:, k] == A[0, k]):
                R[j, k] = np.nan
                continue
            xj = A[:, j] - A[:, j].mean()
            xk = A[:, k] - A[:, k].mean()
            r = np.dot(xj, xk) / np.sqrt(np.dot(xj, xj) * np.dot(xk, xk))
            R[j, k] = min(1.0, max(-1.0, r))
    return R

import numpy as np


def fit_linear_regression(X, y, fit_intercept=True):
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2-D")
    n, d = X.shape  # noqa: RUF059
    if y.ndim != 1 or len(y) != n:
        raise ValueError("y must be 1-D with one entry per row")
    if not (np.isfinite(X).all() and np.isfinite(y).all()):
        raise ValueError("non-finite input")
    A = np.hstack([np.ones((n, 1)), X]) if fit_intercept else X
    if np.linalg.matrix_rank(A) < A.shape[1]:
        raise ValueError("rank deficient")
    beta = np.linalg.solve(A.T @ A, A.T @ y)
    if fit_intercept:
        return float(beta[0]), beta[1:].copy()
    return 0.0, beta

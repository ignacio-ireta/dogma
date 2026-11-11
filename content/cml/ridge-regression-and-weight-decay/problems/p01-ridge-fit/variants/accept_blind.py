import numpy as np


def ridge_fit(X, y, alpha, *, fit_intercept=True):
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X must be a non-empty 2-D array")
    n, d = X.shape
    if y.ndim != 1 or len(y) != n:
        raise ValueError("y must be 1-D with one entry per row")
    if not (np.isfinite(X).all() and np.isfinite(y).all()):
        raise ValueError("non-finite input")
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and >= 0")
    if fit_intercept:
        xm = X.mean(axis=0)
        ym = y.mean()
        Xc = X - xm
        yc = y - ym
    else:
        Xc, yc = X, y
    if alpha == 0 and np.linalg.matrix_rank(Xc) < d:
        raise ValueError("rank deficient at alpha = 0")
    beta = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(d), Xc.T @ yc)
    if fit_intercept:
        return float(ym - xm @ beta), beta
    return 0.0, beta

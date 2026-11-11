import numpy as np


def lasso_cd(X, y, alpha, *, max_iter=10_000, tol=1e-8):
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
    if max_iter < 1:
        raise ValueError("max_iter must be >= 1")
    if not (np.isfinite(tol) and tol > 0):
        raise ValueError("tol must be positive")
    w = np.zeros(d)
    z = (X**2).sum(axis=0) / n
    for _ in range(max_iter):
        biggest = 0.0
        for j in range(d):
            if z[j] == 0:
                continue
            r_j = y - X @ w + X[:, j] * w[j]  # partial residual without coordinate j
            rho = X[:, j] @ r_j / n
            new = (np.sign(rho) * max(abs(rho) - alpha, 0.0)) / z[j]
            biggest = max(biggest, abs(new - w[j]))
            w[j] = new
        if biggest < tol:
            break
    return w

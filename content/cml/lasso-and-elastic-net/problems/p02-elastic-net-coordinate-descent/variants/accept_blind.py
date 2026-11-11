import numpy as np


def elastic_net_cd(X, y, alpha, l1_ratio, *, max_iter=10_000, tol=1e-8):
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
    if not (np.isfinite(l1_ratio) and 0 <= l1_ratio <= 1):
        raise ValueError("l1_ratio must be in [0, 1]")
    if max_iter < 1:
        raise ValueError("max_iter must be >= 1")
    if not (np.isfinite(tol) and tol > 0):
        raise ValueError("tol must be positive")
    w = np.zeros(d)
    z = (X**2).sum(axis=0) / n
    l1 = alpha * l1_ratio
    l2 = alpha * (1 - l1_ratio)
    for _ in range(max_iter):
        biggest = 0.0
        for j in range(d):
            den = z[j] + l2
            if den == 0:
                continue
            r_j = y - X @ w + X[:, j] * w[j]
            rho = X[:, j] @ r_j / n
            new = np.sign(rho) * max(abs(rho) - l1, 0.0) / den
            biggest = max(biggest, abs(new - w[j]))
            w[j] = new
        if biggest < tol:
            break
    return w

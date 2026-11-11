import numpy as np


def fit_logistic_regression(X, y, *, l2=1.0, max_iter=100, tol=1e-10):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X must be a non-empty 2-D array")
    if y.ndim != 1 or len(y) != X.shape[0]:
        raise ValueError("y must be 1-D with one label per row")
    if not np.all(np.isfinite(X)):
        raise ValueError("X must be finite")
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("labels must be 0 or 1")
    if np.all(y == y[0]):
        raise ValueError("both classes are needed")
    if not (np.isfinite(l2) and l2 >= 0):
        raise ValueError("l2 must be finite and >= 0")
    if int(max_iter) != max_iter or max_iter < 1 or isinstance(max_iter, bool):
        raise ValueError("max_iter must be an integer >= 1")
    if not (np.isfinite(tol) and tol > 0):
        raise ValueError("tol must be positive")
    n, d = X.shape
    A = np.hstack([np.ones((n, 1)), X])
    pen = np.r_[0.0, np.full(d, float(l2))]

    def sig(z):
        # Logistic function via tanh: no exponential is ever formed explicitly.
        return 0.5 * (1.0 + np.tanh(0.5 * z))

    def loss(theta):
        eta = A @ theta
        return np.mean(np.logaddexp(0.0, eta) - y * eta) + 0.5 * np.sum(pen * theta**2)

    def grad(theta):
        return A.T @ (sig(A @ theta) - y) / n + pen * theta

    theta = np.zeros(d + 1)
    for _ in range(max_iter):
        g = grad(theta)
        if np.max(np.abs(g)) <= tol:
            return float(theta[0]), theta[1:].copy()
        p = sig(A @ theta)
        H = (A * (p * (1 - p))[:, None]).T @ A / n + np.diag(pen)
        step = np.linalg.solve(H, -g)
        t = 1.0
        f0 = loss(theta)
        slope = float(g @ step)  # negative: Newton is a descent direction
        while t > 1e-14 and loss(theta + t * step) > f0 + 1e-4 * t * slope + 1e-15 * abs(f0):
            t *= 0.5
        theta = theta + t * step
    g = grad(theta)
    if np.max(np.abs(g)) <= tol:
        return float(theta[0]), theta[1:].copy()
    raise RuntimeError("did not converge")

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

_ARMIJO = 0.0001
_MIN_STEP = 1e-12


def _softplus(eta: NDArray[np.float64]) -> NDArray[np.float64]:
    """log(1 + exp(eta)) without overflow: max(eta, 0) + log1p(exp(-|eta|))."""
    return np.maximum(eta, 0.0) + np.log1p(np.exp(-np.abs(eta)))


def _sigmoid(eta: NDArray[np.float64]) -> NDArray[np.float64]:
    """The stable logistic function, built from exp(-|eta|) so no exponent is positive."""
    e = np.exp(-np.abs(eta))
    return np.where(eta >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


def _validate(
    X: ArrayLike, y: ArrayLike, l2: float, max_iter: int, tol: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    X_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if X_arr.ndim != 2 or X_arr.shape[0] == 0 or X_arr.shape[1] == 0:
        raise ValueError("X must be a non-empty 2-D array with one sample per row")
    if y_arr.ndim != 1 or y_arr.shape[0] != X_arr.shape[0]:
        raise ValueError("y must be 1-D with one label per row of X")
    if not np.all(np.isfinite(X_arr)):
        raise ValueError("X must contain only finite numbers")
    if not np.all((y_arr == 0.0) | (y_arr == 1.0)):
        raise ValueError("y must contain only the labels 0 and 1")
    if np.all(y_arr == y_arr[1]):
        raise ValueError("y must contain both classes; a single class has no finite fit")
    if not (np.isfinite(l2) and l2 >= 0.0):
        raise ValueError("l2 must be a finite number that is zero or positive")
    if isinstance(max_iter, bool) or not isinstance(max_iter, int | np.integer) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    if not (np.isfinite(tol) and tol > 0.0):
        raise ValueError("tol must be a positive finite number")
    return (X_arr, y_arr)


def fit_logistic_regression(
    X: ArrayLike, y: ArrayLike, *, l2: float = 1.0, max_iter: int = 100, tol: float = 1e-10
) -> tuple[float, NDArray[np.float64]]:
    """Return (intercept, coef) minimising mean log-loss + (l2 / 2) * ||coef||^2.

    Newton's method on the convex objective J(theta), with an intercept column first and a
    diagonal penalty that is zero on the intercept. Each step is damped by backtracking until
    the objective decreases (Armijo condition). The stopping rule is max |grad J| <= tol.
    """
    X_arr, y_arr = _validate(X, y, l2, max_iter, tol)
    n, d = X_arr.shape
    A = np.column_stack([np.ones(n), X_arr])
    penalty = np.full(d + 1, l2)
    penalty[0] = 0.0

    def objective(theta: NDArray[np.float64]) -> float:
        eta = A @ theta
        return float(np.mean(_softplus(eta) - y_arr * eta) + 0.5 * np.sum(penalty * theta**2))

    def gradient(theta: NDArray[np.float64]) -> NDArray[np.float64]:
        return A.T @ (_sigmoid(A @ theta) - y_arr) / n + penalty * theta

    theta = np.zeros(d + 1)
    for _ in range(max_iter):
        grad = gradient(theta)
        if np.max(np.abs(grad)) <= tol:
            return (float(theta[0]), theta[1:].copy())
        p = _sigmoid(A @ theta)
        weights = p * (1.0 - p)
        hessian = A.T * weights @ A / n + np.diag(penalty)
        try:
            step = np.linalg.solve(hessian, -grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, -grad, rcond=None)[0]
        f0 = objective(theta)
        slope = float(grad @ step)
        slack = 16.0 * np.finfo(np.float64).eps * max(1.0, abs(f0))
        t = 1.0
        while t > _MIN_STEP and objective(theta + t * step) > f0 + _ARMIJO * t * slope + slack:
            t *= 0.5
        theta = theta + t * step
    grad = gradient(theta)
    if np.max(np.abs(grad)) <= tol:
        return (float(theta[0]), theta[1:].copy())
    raise RuntimeError(
        f"logistic regression did not reach the gradient tolerance in {max_iter} iterations"
    )

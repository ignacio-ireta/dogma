from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def lasso_cd(
    X: ArrayLike,
    y: ArrayLike,
    alpha: float,
    *,
    max_iter: int = 10_000,
    tol: float = 1e-8,
) -> NDArray[np.float64]:
    """Wrong on purpose: accepts max_iter equal to zero instead of raising.

    Minimises (1/(2n)) ||y - X w||^2 + alpha ||w||_1. For each coordinate j the update is
    w_j = S(rho_j, alpha) / z_j, with rho_j = (1/n) X_j^T (y - sum_{k != j} X_k w_k),
    z_j = (1/n) ||X_j||^2, and S(x, lam) = sign(x) * max(|x| - lam, 0). A zero column keeps w_j = 0.
    Stops after a sweep whose largest coefficient change is below tol, or after max_iter sweeps.

    Raises
    ------
    ValueError
        If X is not 2-D, y is not 1-D with one entry per row, the data is empty or non-finite,
        alpha is negative or non-finite, max_iter < 1, or tol is not a positive number.
    """
    # np.array copies, so the caller's arrays are never written to.
    x_arr = np.array(X, dtype=np.float64)
    y_arr = np.array(y, dtype=np.float64)
    if x_arr.ndim != 2 or x_arr.shape[0] == 0 or x_arr.shape[1] == 0:
        raise ValueError(f"X must be 2-D with at least one row and column, got shape {x_arr.shape}")
    n, d = x_arr.shape
    if y_arr.ndim != 1 or y_arr.shape[0] != n:
        raise ValueError(f"y must be 1-D with {n} entries, got shape {y_arr.shape}")
    if not (np.all(np.isfinite(x_arr)) and np.all(np.isfinite(y_arr))):
        raise ValueError("X and y must contain only finite numbers")
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError(f"alpha must be a finite number >= 0, got {alpha!r}")
    if max_iter < 0:
        raise ValueError(f"max_iter must be at least 1, got {max_iter!r}")
    if not (np.isfinite(tol) and tol > 0):
        raise ValueError(f"tol must be a finite number greater than 0, got {tol!r}")

    # z_j = (1/n) ||X_j||^2 is the curvature of the objective along coordinate j.
    z = np.sum(x_arr**2, axis=0) / n
    w = np.zeros(d)
    # Keep the residual r = y - X w up to date, so each coordinate costs O(n) instead of O(nd).
    residual = y_arr.copy()

    for _ in range(max_iter):
        largest_change = 0.0
        for j in range(d):
            if z[j] == 0.0:
                # An all-zero column has no effect on the fit, so its coefficient stays at 0.
                continue
            column = x_arr[:, j]
            # Partial residual: add back the current contribution of coordinate j.
            residual += column * w[j]
            # rho_j: correlation of column j with what the other coordinates leave unexplained.
            rho = column @ residual / n
            # Soft-threshold: a correlation no larger than alpha in size gives exactly zero, and
            # a larger one is shrunk toward zero by alpha. Dividing by z_j undoes the column scale.
            new_w = np.sign(rho) * max(abs(rho) - alpha, 0.0) / z[j]
            residual -= column * new_w
            largest_change = max(largest_change, abs(new_w - w[j]))
            w[j] = new_w
        if largest_change < tol:
            break
    return w

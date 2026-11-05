from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def fit_linear_regression(
    X: ArrayLike,
    y: ArrayLike,
    fit_intercept: bool = True,
) -> tuple[float, NDArray[np.float64]]:
    """Least-squares fit of y ~ intercept + X @ coef by the normal equations.

    Minimises ||y - A beta||^2 where A is the design matrix: [1 | X] when fit_intercept, else X.
    The minimiser satisfies (A^T A) beta = A^T y.

    Returns
    -------
    (intercept, coef) : intercept is a Python float (0.0 when fit_intercept is False);
        coef is a (d,) float64 array.
    """
    # Copies as float64: the caller's arrays are never written to, and integer data works.
    X_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if X_arr.ndim != 2:
        raise ValueError(f"X must be 2-D (n, d), got shape {X_arr.shape}")
    n = X_arr.shape[0]
    if y_arr.ndim != 1 or y_arr.shape[0] != n:
        raise ValueError(f"y must be 1-D with {n} entries, got shape {y_arr.shape}")
    if not (np.all(np.isfinite(X_arr)) and np.all(np.isfinite(y_arr))):
        raise ValueError("X and y must contain only finite numbers")

    design = np.hstack([np.ones((n, 1)), X_arr]) if fit_intercept else X_arr
    # The normal equations have a unique solution only when the design has full column rank.
    # Check the design itself, not A^T A: squaring the matrix would square its condition number.
    if np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError(
            "design matrix is rank deficient: a column is a combination of the others "
            "(or the intercept column duplicates a constant feature)"
        )

    # Setting the gradient of ||y - A beta||^2 to zero gives (A^T A) beta = A^T y. Solve that
    # system directly; forming an explicit inverse would be slower and less accurate.
    beta = np.linalg.solve(design.T @ design, design.T @ y_arr)
    if fit_intercept:
        return float(beta[0]), beta[1:].copy()
    return 0.0, beta

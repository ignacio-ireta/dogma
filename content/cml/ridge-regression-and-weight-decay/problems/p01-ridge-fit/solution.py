from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def ridge_fit(
    X: ArrayLike,
    y: ArrayLike,
    alpha: float,
    *,
    fit_intercept: bool = True,
) -> tuple[float, NDArray[np.float64]]:
    """Ridge regression: minimise ||y - b0 - X beta||^2 + alpha * ||beta||^2.

    The intercept b0 is not penalised. With fit_intercept, X and y are centred first, so the
    coefficients solve (Xc^T Xc + alpha I) beta = Xc^T yc, and b0 = mean(y) - mean(X) @ beta.
    Without it, beta solves (X^T X + alpha I) beta = X^T y and the intercept is 0.0.

    Returns
    -------
    (intercept, coef) : intercept is a Python float; coef is a (d,) float64 array.

    Raises
    ------
    ValueError
        If X is not a nonempty 2-D array, y is not 1-D with one entry per row, any value is
        non-finite, alpha is negative, or alpha is 0 and the design (centred when
        fit_intercept) is rank deficient.
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

    if fit_intercept:
        # Centring removes the intercept from the problem, so the penalty never touches it.
        x_mean = x_arr.mean(axis=0)
        y_mean = float(y_arr.mean())
        xc = x_arr - x_mean
        yc = y_arr - y_mean
    else:
        xc, yc = x_arr, y_arr

    # With alpha = 0 the system is the plain normal equations, which need full column rank.
    # With alpha > 0 the matrix Xc^T Xc + alpha I is positive definite for any Xc.
    if alpha == 0 and np.linalg.matrix_rank(xc) < d:
        raise ValueError(
            "alpha = 0 needs a full-rank design (centred when fit_intercept): a column is a "
            "combination of the others, so the least-squares coefficients are not unique"
        )

    # The penalty adds alpha to the diagonal of the Gram matrix. It shrinks every coefficient in the
    # directions where X has little variance, which is what makes the rank-deficient case solvable.
    gram = xc.T @ xc + alpha * np.eye(d)
    beta = np.linalg.solve(gram, xc.T @ yc)
    if fit_intercept:
        # Undo the centring: the line passes through the point of means (x_mean, y_mean).
        return float(y_mean - x_mean @ beta), beta
    return 0.0, beta

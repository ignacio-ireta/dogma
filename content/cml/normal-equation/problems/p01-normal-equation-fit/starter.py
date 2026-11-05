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

    Raises
    ------
    ValueError
        If X is not 2-D, y is not 1-D with one entry per row, any value is non-finite, or the
        design matrix is rank deficient (rank below its number of columns).
    """
    raise NotImplementedError

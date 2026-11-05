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

    The intercept b0 is not penalised. With fit_intercept, centre X and y first and solve
    (Xc^T Xc + alpha I) beta = Xc^T yc, then b0 = mean(y) - mean(X) @ beta. Without it, solve
    (X^T X + alpha I) beta = X^T y and return intercept 0.0.

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
    raise NotImplementedError

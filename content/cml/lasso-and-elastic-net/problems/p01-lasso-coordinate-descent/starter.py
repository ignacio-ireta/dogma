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
    """Lasso coefficients by cyclic coordinate descent, with no intercept.

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
    raise NotImplementedError

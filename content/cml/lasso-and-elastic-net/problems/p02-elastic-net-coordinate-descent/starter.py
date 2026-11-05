from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def elastic_net_cd(
    X: ArrayLike,
    y: ArrayLike,
    alpha: float,
    l1_ratio: float,
    *,
    max_iter: int = 10_000,
    tol: float = 1e-8,
) -> NDArray[np.float64]:
    """Elastic net coefficients by cyclic coordinate descent, with no intercept.

    Minimises (1/(2n)) ||y - X w||^2 + alpha * l1_ratio * ||w||_1
    + 0.5 * alpha * (1 - l1_ratio) * ||w||^2.
    For coordinate j, with rho_j = (1/n) X_j^T (y - sum_{k != j} X_k w_k) and z_j = (1/n) ||X_j||^2,
    the update is w_j = S(rho_j, alpha * l1_ratio) / (z_j + alpha * (1 - l1_ratio)), where
    S(x, lam) = sign(x) * max(|x| - lam, 0). A zero denominator keeps w_j = 0.

    Raises
    ------
    ValueError
        If X is not 2-D, y is not 1-D with one entry per row, the data is empty or non-finite,
        alpha is negative or non-finite, l1_ratio is outside [0, 1], max_iter < 1, or tol is not
        positive.
    """
    raise NotImplementedError

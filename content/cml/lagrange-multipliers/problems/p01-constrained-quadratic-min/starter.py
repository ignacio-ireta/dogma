from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def constrained_quadratic_min(
    Q: ArrayLike, c: ArrayLike, A: ArrayLike, b: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Minimise 1/2 x'Qx + c'x subject to Ax = b, returning (x, lam).

    Use the Lagrangian f(x) - lam'(Ax - b), so lam is the derivative of the optimal value
    with respect to b. Solve the KKT system for x and lam together.

    Parameters
    ----------
    Q : (n, n) symmetric positive definite matrix.
    c : (n,) linear term.
    A : (m, n) constraint matrix with m >= 1 and full row rank.
    b : (m,) constraint levels.

    Returns
    -------
    (x, lam): x has shape (n,), lam has shape (m,), both float64.

    Raises
    ------
    ValueError
        On bad shapes, non-finite entries, an asymmetric or non-positive-definite Q, or a
        rank-deficient A.
    """
    raise NotImplementedError

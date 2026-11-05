from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def fit_logistic_regression(
    X: ArrayLike,
    y: ArrayLike,
    *,
    l2: float = 1.0,
    max_iter: int = 100,
    tol: float = 1e-10,
) -> tuple[float, NDArray[np.float64]]:
    """Return (intercept, coef) minimising mean log-loss + (l2 / 2) * ||coef||^2.

    The intercept is not penalised. The loss uses labels 0 and 1; the gradient at the returned
    point has largest absolute entry at most `tol`.

    Parameters
    ----------
    X : (n, d) array-like of finite values, n >= 1, d >= 1.
    y : (n,) array-like with every entry 0 or 1, containing both classes.
    l2 : finite number >= 0 (zero is allowed).
    max_iter : integer >= 1, the iteration budget.
    tol : finite number > 0, the gradient tolerance.

    Returns
    -------
    (intercept, coef): a float and a float64 array of shape (d,).

    Raises
    ------
    ValueError
        For bad shapes, labels, a single class, non-finite values, or invalid l2, max_iter, tol.
    RuntimeError
        When max_iter iterations do not reach the gradient tolerance.
    """
    raise NotImplementedError

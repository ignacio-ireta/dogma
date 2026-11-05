from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def covariance_matrix(X: ArrayLike, ddof: int = 1) -> NDArray[np.float64]:
    """Return the (d, d) sample covariance matrix of the (n, d) data matrix ``X``.

    Rows are observations and columns are variables. The result is
    (X - mean)^T (X - mean) / (n - ddof), with the column means subtracted first.

    Parameters
    ----------
    X : (n, d) array-like, n >= 1 rows and d >= 1 columns, all entries finite (ints allowed).
    ddof : non-negative integer; the divisor is n - ddof (default 1).

    Returns
    -------
    (d, d) float64 array.

    Raises
    ------
    ValueError
        If X is not 2-D, has no rows or no columns, contains NaN or infinity, ddof is negative
        or not an integer, or n <= ddof.
    """
    raise NotImplementedError

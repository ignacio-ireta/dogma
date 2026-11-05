from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def correlation_matrix(X: ArrayLike) -> NDArray[np.float64]:
    """Return the (d, d) Pearson correlation matrix of the (n, d) data matrix ``X``.

    Rows are observations and columns are variables. Entry [j, k] is
    C[j, k] / sqrt(C[j, j] * C[k, k]), where C is the sample covariance with divisor n - 1.

    Parameters
    ----------
    X : (n, d) array-like, n >= 2 rows and d >= 1 columns, all entries finite (ints allowed).

    Returns
    -------
    (d, d) float64 array with every entry in [-1, 1]. For a constant column (all entries equal),
    its entire row and column are NaN, including the diagonal.

    Raises
    ------
    ValueError
        If X is not 2-D, has no rows or no columns, has fewer than two rows, or contains NaN or
        infinity.
    """
    raise NotImplementedError

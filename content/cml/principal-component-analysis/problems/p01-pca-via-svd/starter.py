from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def pca(X: ArrayLike, n_components: int) -> dict[str, NDArray[np.float64]]:
    """Return the top ``n_components`` principal components of the (n, d) data matrix ``X``.

    Rows are observations and columns are variables. Centre each column, take the SVD of the
    centred data, and return a dict with the keys "components" (k, d), "explained_variance" (k,),
    "explained_variance_ratio" (k,) and "transformed" (n, k), where k = n_components.

    Explained variances use the divisor n - 1. Ratios divide by the total variance of all d columns,
    and are 0.0 when that total is zero. Each row of "components" has its largest-magnitude entry
    positive; an exact tie goes to the lowest index. Raise ValueError for invalid input (not 2-D,
    fewer than 2 rows, no columns, non-finite entries, or n_components not an integer in
    [1, min(n, d)]). Do not mutate X.

    Parameters
    ----------
    X : (n, d) real array-like, n >= 2, d >= 1, finite (integers allowed).
    n_components : int, 1 <= n_components <= min(n, d). Booleans are rejected.

    Returns
    -------
    dict with keys "components", "explained_variance", "explained_variance_ratio", "transformed".
    """
    raise NotImplementedError

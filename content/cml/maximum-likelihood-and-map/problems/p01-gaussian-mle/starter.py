from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def gaussian_mle(X: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the maximum-likelihood mean and covariance of a multivariate normal.

    Parameters
    ----------
    X : (n, d) array-like, n >= 1 rows (observations) and d >= 1 columns (variables), all finite.

    Returns
    -------
    (mean, covariance) : tuple of float64 arrays with shapes (d,) and (d, d).
        mean = column means. covariance = (1/n) * sum of (x - mean) (x - mean)^T, dividing by n.

    Raises
    ------
    ValueError
        If X is not 2-D, has no rows or no columns, or contains NaN or infinity.
    """
    raise NotImplementedError

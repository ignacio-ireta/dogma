from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def mvn_logpdf(X: ArrayLike, mean: ArrayLike, cov: ArrayLike) -> NDArray[np.float64]:
    """Return the log-density of N(mean, cov) at each row of ``X``.

    Parameters
    ----------
    X : (n, d) array-like, n >= 1 rows and d >= 1 columns, all finite (ints allowed).
    mean : (d,) array-like, finite.
    cov : (d, d) array-like, symmetric and positive definite.

    Returns
    -------
    (n,) float64 array with
    -d/2 * log(2 pi) - 1/2 * log det(cov) - 1/2 * (x - mean)^T cov^{-1} (x - mean).
    Compute it with a Cholesky factorisation, never by forming the inverse or the determinant.

    Raises
    ------
    ValueError
        If X is not 2-D or is empty, mean does not have length d, cov is not (d, d), an input
        contains NaN or infinity, cov is not symmetric, or cov is not positive definite.
    """
    raise NotImplementedError

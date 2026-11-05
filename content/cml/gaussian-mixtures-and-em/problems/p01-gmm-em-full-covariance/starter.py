from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def gmm_em(
    X: ArrayLike,
    k: int,
    init_means: ArrayLike,
    n_iter: int,
    reg: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Fit a full-covariance Gaussian mixture with EM and a covariance floor.

    Parameters
    ----------
    X : (n, d) finite samples, n >= 1, d >= 1.
    k : integer component count, 1 <= k <= n.
    init_means : (k, d) finite starting means.
    n_iter : integer >= 0 number of EM iterations (booleans are rejected).
    reg : finite real > 0, the covariance floor (every eigenvalue of every covariance >= reg).

    Returns
    -------
    (weights, means, covs, lls): float64 arrays with shapes (k,), (k, d), (k, d, d) and
    (n_iter + 1,). lls[t] is the log-likelihood after t iterations.

    Raises
    ------
    ValueError
        On bad shapes, non-finite values, k outside [1, n], n_iter < 0, or reg <= 0.
    """
    raise NotImplementedError

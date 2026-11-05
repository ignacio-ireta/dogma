from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def gaussian_mle(X: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the maximum-likelihood mean and covariance of a multivariate normal.

    Setting the derivatives of the Gaussian log-likelihood to zero gives the sample mean and the
    average outer product of the deviations from it. The divisor is n, not n - 1.
    """
    x = np.asarray(X, dtype=np.float64)

    if x.ndim != 2:
        raise ValueError(f"expected a 2-D array (observations, variables), got {x.ndim}-D")
    n, d = x.shape
    if n == 0 or d == 0:
        raise ValueError("need at least one observation and one variable")
    if not np.isfinite(x).all():
        raise ValueError("observations must be finite (no NaN or inf)")

    # The MLE of the mean is the sample mean.
    mean = x.mean(axis=0)
    # Centre first. Each deviation is small, so the products do not cancel.
    centred = x - mean
    # The MLE of the covariance averages the outer products of the deviations over n. That
    # is biased by the factor (n - 1) / n, which the lesson explains.
    covariance = centred.T @ centred / n
    return mean, covariance

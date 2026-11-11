from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral

import numpy as np
from numpy.typing import NDArray


def acf(x: Sequence[float] | NDArray[np.float64], nlags: int) -> NDArray[np.float64]:
    """Wrong on purpose: sample autocorrelation of x at lags 0 .. nlags.

    With xbar the sample mean and d_t = x_t - xbar:

        r_k = sum_{t=0}^{n-k-1} d_t * d_{t+k}  /  sum_{t=0}^{n-1} d_t^2

    Every lag divides by the same n * var, not by the (n - k) pairs that are actually summed.
    That is the biased estimator (statsmodels' ``adjusted=False``). r_0 is exactly 1.

    Returns
    -------
    r : (nlags + 1,) float64 array, r[k] is the autocorrelation at lag k.
    """
    values = np.asarray(x, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("x must be one-dimensional")
    n = values.size
    if n < 2:
        raise ValueError("x needs at least two values")
    if not np.all(np.isfinite(values)):
        raise ValueError("x must contain only finite numbers")
    if isinstance(nlags, bool) or not isinstance(nlags, Integral) or not 0 <= nlags <= n - 1:
        raise ValueError("nlags must be an integer from 0 to len(x) - 1")
    # An exactly constant series has zero variance, so the ratio is undefined. Test exact equality
    # rather than a small denominator: a constant like 0.1 has a mean that is not exactly 0.1.
    if np.all(values == values[0]):
        raise ValueError("x is constant, so its autocorrelation is undefined")

    # Subtract an observed anchor before taking the mean, preserving a small spread on a
    # large common offset. Rescale before products: the ACF is unchanged by any nonzero scale.
    with np.errstate(over="ignore"):
        d = values - values[0]
    if not np.all(np.isfinite(d)):
        # Opposite signs near float64's limit can overflow even the anchored subtraction.
        d = values / np.max(np.abs(values))
    d = d / np.max(np.abs(d))
    d = d - np.floor(d.mean())
    denominator = float(np.dot(d, d))  # n * biased variance, shared by every lag
    out = np.empty(nlags + 1, dtype=np.float64)
    for k in range(nlags + 1):
        # Sum the n - k products that exist at lag k, but divide by the full denominator.
        out[k] = np.dot(d[: n - k], d[k:]) / denominator
    return out

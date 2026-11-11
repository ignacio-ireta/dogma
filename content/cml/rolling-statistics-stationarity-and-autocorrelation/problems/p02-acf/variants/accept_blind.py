import numbers

import numpy as np


def acf(x, nlags):
    values = np.asarray(x, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("x must be one-dimensional")
    n = values.size
    if n < 2:
        raise ValueError("x needs at least two values")
    if not np.all(np.isfinite(values)):
        raise ValueError("x must contain only finite numbers")
    if (
        isinstance(nlags, bool)
        or not isinstance(nlags, numbers.Integral)
        or not 0 <= nlags <= n - 1
    ):
        raise ValueError("nlags must be an integer from 0 to len(x) - 1")
    if np.all(values == values[0]):
        raise ValueError("x is constant, so its autocorrelation is undefined")
    # Scale first so squares cannot overflow or underflow; the ACF does not change under scaling.
    scaled = values / np.max(np.abs(values))
    d = scaled - scaled.mean()
    total = float(np.sum(d * d))
    out = np.empty(nlags + 1)
    for k in range(nlags + 1):
        # every lag shares the total sum of squares as its divisor
        out[k] = np.sum(d[: n - k] * d[k:]) / total
    return out

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

# math.lgamma works on scalars; vectorise it once so it applies elementwise to arrays of k.
_lgamma = np.vectorize(math.lgamma, otypes=[np.float64])


def binomial_pmf(k: ArrayLike, n: int, p: float) -> NDArray[np.float64]:
    """Wrong on purpose: return P(X = k) for X ~ Binomial(n, p), elementwise over ``k``, computed in log space."""  # noqa: E501
    # n must be a non-negative integer. int(n) != n rejects 2.5 and -1 alike, since -1 is an int.
    n_int = int(n)
    if n_int != n or n_int < 0:
        raise ValueError(f"n must be a non-negative integer, got {n!r}")
    # A positive test, so that NaN is rejected too: NaN fails every comparison.
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1], got {p!r}")

    # k may be ints, or floats with integral values. Anything else is a bug in the caller.
    k_arr = np.asarray(k)
    if k_arr.dtype.kind == "f":
        if not (np.isfinite(k_arr).all() and (k_arr == np.floor(k_arr)).all() and k_arr.ndim == 0):
            raise ValueError("k must contain integers (integral floats are accepted)")
    elif k_arr.dtype.kind not in "iu":
        raise ValueError("k must contain integers")
    k_int = k_arr.astype(np.int64)

    out = np.zeros(k_int.shape, dtype=np.float64)

    # log(0) is -inf, so p = 0 and p = 1 are answered exactly, not through the logarithms below.
    if p == 0.0:
        out[...] = k_int == 0
        return out
    if p == 1.0:
        out[...] = k_int == n_int
        return out

    # Outside [0, n] the probability is exactly 0. lgamma of a negative integer is undefined,
    # so only the values inside the support go through the formula.
    inside = (k_int >= 0) & (k_int <= n_int)
    kk = k_int[inside]

    # log C(n, k) = lgamma(n+1) - lgamma(k+1) - lgamma(n-k+1). The coefficient itself never
    # exists as a float, so it cannot overflow. Only its logarithm is formed.
    log_choose = math.lgamma(n_int + 1) - _lgamma(kk + 1) - _lgamma(n_int - kk + 1)
    # log P = log C(n, k) + k log p + (n - k) log(1 - p); log1p keeps 1 - p accurate near p = 0.
    log_pmf = log_choose + kk * math.log(p) + (n_int - kk) * math.log1p(-p)
    out[inside] = np.exp(log_pmf)
    return out

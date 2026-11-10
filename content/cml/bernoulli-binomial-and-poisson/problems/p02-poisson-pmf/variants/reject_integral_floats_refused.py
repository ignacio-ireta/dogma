from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

# math.lgamma works on scalars; vectorise it once so it applies elementwise to arrays of k.
_lgamma = np.vectorize(math.lgamma, otypes=[np.float64])


def poisson_pmf(k: ArrayLike, lam: float) -> NDArray[np.float64]:
    """Wrong on purpose: return P(X = k) for X ~ Poisson(lam), elementwise over ``k``, computed in log space."""  # noqa: E501
    # A positive test: negative, NaN and infinite rates all fail it.
    lam_f = float(lam)
    if not 0.0 <= lam_f < math.inf:
        raise ValueError(f"lam must be a finite non-negative rate, got {lam!r}")

    # k may be ints, or floats with integral values. Anything else is a bug in the caller.
    k_arr = np.asarray(k)
    if k_arr.dtype.kind == "f":
        if not (np.isfinite(k_arr).all() and (k_arr == np.floor(k_arr)).all() and k_arr.ndim == 0):
            raise ValueError("k must contain integers (integral floats are accepted)")
    elif k_arr.dtype.kind not in "iu":
        raise ValueError("k must contain integers")
    k_int = k_arr.astype(np.int64)

    out = np.zeros(k_int.shape, dtype=np.float64)

    # A rate of 0 means no events: log(0) is -inf, so this case is answered exactly.
    if lam_f == 0.0:
        out[...] = k_int == 0
        return out

    # Counts below 0 have probability 0. Only k >= 0 goes through the formula.
    inside = k_int >= 0
    kk = k_int[inside]
    # log P(X = k) = k log(lam) - lam - log(k!), and log(k!) = lgamma(k + 1). The factorial itself
    # overflows float64 at k = 171, so it is never formed.
    log_pmf = kk * math.log(lam_f) - lam_f - _lgamma(kk + 1)
    out[inside] = np.exp(log_pmf)
    return out

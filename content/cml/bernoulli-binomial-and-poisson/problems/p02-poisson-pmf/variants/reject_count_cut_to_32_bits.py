from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

_lgamma = np.vectorize(math.lgamma, otypes=[np.float64])


def poisson_pmf(k: ArrayLike, lam: float) -> NDArray[np.float64]:
    """Return P(X = k) for X ~ Poisson(lam), elementwise over ``k``, computed in log space."""
    lam_f = float(lam)
    if not 0.0 <= lam_f < math.inf:
        raise ValueError(f"lam must be a finite non-negative rate, got {lam!r}")
    k_arr = np.asarray(k)
    if k_arr.dtype.kind == "f":
        if not (np.isfinite(k_arr).all() and (k_arr == np.floor(k_arr)).all()):
            raise ValueError("k must contain integers (integral floats are accepted)")
    elif k_arr.dtype.kind not in "iu":
        raise ValueError("k must contain integers")
    k_int = k_arr.astype(np.int32)
    out = np.zeros(k_int.shape, dtype=np.float64)
    if lam_f == 0.0:
        out[...] = k_int == 0
        return out
    inside = k_int >= 0
    kk = k_int[inside]
    log_pmf = kk * math.log(lam_f) - lam_f - _lgamma(kk + 1)
    out[inside] = np.exp(log_pmf)
    return out

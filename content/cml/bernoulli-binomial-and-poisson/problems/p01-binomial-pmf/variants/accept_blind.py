from __future__ import annotations

import math
import numbers

import numpy as np
from numpy.typing import ArrayLike, NDArray


def binomial_pmf(k: ArrayLike, n: int, p: float) -> NDArray[np.float64]:
    """Return P(X = k) for X ~ Binomial(n, p), elementwise over ``k``, computed in log space."""
    if isinstance(n, bool) or not isinstance(n, (numbers.Integral, np.integer)) or n < 0:
        raise ValueError("n must be a non-negative integer")
    n = int(n)
    try:
        p_f = float(p)
    except (TypeError, ValueError) as exc:
        raise ValueError("p must be a real number in [0, 1]") from exc
    if not (0.0 <= p_f <= 1.0):  # NaN fails this comparison too
        raise ValueError("p must be in [0, 1]")

    karr = np.asarray(k)
    if karr.dtype.kind not in "biuf":
        raise ValueError("k must contain integers")
    kf = karr.astype(np.float64)
    if not np.all(np.isfinite(kf)) or np.any(kf != np.floor(kf)):
        raise ValueError("k must contain integers")

    valid = (kf >= 0) & (kf <= n)
    kk = np.where(valid, kf, 0.0).astype(np.int64)

    if p_f == 0.0:
        return np.asarray(np.where(valid & (kk == 0), 1.0, 0.0), dtype=np.float64)
    if p_f == 1.0:
        return np.asarray(np.where(valid & (kk == n), 1.0, 0.0), dtype=np.float64)

    log_p = math.log(p_f)
    log_q = math.log1p(-p_f)
    log_n_fact = math.lgamma(n + 1)
    flat = kk.ravel()
    logs = np.array(
        [
            log_n_fact
            - math.lgamma(int(x) + 1)
            - math.lgamma(n - int(x) + 1)
            + int(x) * log_p
            + (n - int(x)) * log_q
            for x in flat
        ],
        dtype=np.float64,
    ).reshape(kk.shape)
    out = np.where(valid, np.exp(logs), 0.0)
    return np.asarray(out, dtype=np.float64)

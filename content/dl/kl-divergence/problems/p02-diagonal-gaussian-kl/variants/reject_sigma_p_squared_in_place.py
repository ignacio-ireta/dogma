from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _vector(value: ArrayLike, name: str, *, positive: bool) -> NDArray[np.float64]:
    """Validate a 1-D real vector. Standard deviations must be strictly positive."""
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be real numeric (bool, complex and object are rejected)")
    vector = np.asarray(raw, dtype=np.float64)
    if vector.ndim != 1 or vector.size < 1:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be finite")
    if positive and not np.all(vector > 0):
        raise ValueError(f"{name} must be strictly positive: a standard deviation is never zero")
    return vector


def diagonal_gaussian_kl(
    mu_p: ArrayLike, sigma_p: ArrayLike, mu_q: ArrayLike, sigma_q: ArrayLike
) -> float:
    """Wrong on purpose: sigma_p is squared in place in the caller's array.

    Each argument is a 1-D array of length d. Closed form, summed over the d independent
    coordinates: sum_i [ log(sq_i / sp_i) + (sp_i^2 + (mp_i - mq_i)^2) / (2 sq_i^2) - 1/2 ].
    """
    mp = _vector(mu_p, "mu_p", positive=False)
    sp = _vector(sigma_p, "sigma_p", positive=True)
    mq = _vector(mu_q, "mu_q", positive=False)
    sq = _vector(sigma_q, "sigma_q", positive=True)
    if not (mp.shape == sp.shape == mq.shape == sq.shape):
        raise ValueError("all four vectors must have the same length d")

    # The log term is positive when q is wider than p and negative when it is narrower.
    log_scale = np.log(sq) - np.log(sp)
    # The spread term is the expected squared deviation of P under Q's variance, divided by
    # 2 sq^2. The sp^2 term is P's own variance; (mp - mq)^2 is the mean shift.
    sp **= 2
    spread = (sp + (mp - mq) ** 2) / (2.0 * sq**2)
    # The -1/2 is the constant that makes identical distributions give exactly 0.
    return float(np.sum(log_scale + spread - 0.5))

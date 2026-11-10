from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike


def _prepare(name: str, values: ArrayLike) -> np.ndarray:
    arr = np.asarray(values)
    if arr.dtype.kind not in ("i", "u", "f"):
        raise ValueError(f"{name} must hold real numeric values, got dtype {arr.dtype}")
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    arr = arr.astype(np.float64)
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must contain only finite entries")
    return arr


def diagonal_gaussian_kl(
    mu_p: ArrayLike, sigma_p: ArrayLike, mu_q: ArrayLike, sigma_q: ArrayLike
) -> float:
    """Return KL(N(mu_p, diag(sigma_p^2)) || N(mu_q, diag(sigma_q^2))) in nats.

    Inputs are 1-D real arrays of one common length d >= 1 with finite entries. Every sigma
    entry must be strictly positive. Closed form, summed over coordinates:
    sum_i [ log(sq_i / sp_i) + (sp_i^2 + (mp_i - mq_i)^2) / (2 sq_i^2) - 1/2 ].
    Returns a Python float. Raise ValueError for invalid input. Do not mutate inputs.
    """
    mp = _prepare("mu_p", mu_p)
    sp = _prepare("sigma_p", sigma_p)
    mq = _prepare("mu_q", mu_q)
    sq = _prepare("sigma_q", sigma_q)
    lengths = {mp.shape[0], sp.shape[0], mq.shape[0], sq.shape[0]}
    if len(lengths) != 1:
        raise ValueError("mu_p, sigma_p, mu_q and sigma_q must all have the same length")
    if not ((sp > 0).all() and (sq > 0).all()):
        raise ValueError("every standard deviation must be strictly positive")

    with np.errstate(over="raise", under="ignore"):
        # Ratios first, so that squares stay within float64 range for sigmas in [1e-50, 1e50].
        log_scale = np.log(sq) - np.log(sp)
        spread = (sp / sq) ** 2
        mean_term = ((mp - mq) / sq) ** 2
        terms = log_scale + 0.5 * (spread + mean_term) - 0.5
    return float(math.fsum(terms.tolist()))

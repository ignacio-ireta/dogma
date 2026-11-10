from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats


def mean_confidence_interval(x: ArrayLike, confidence: float = 0.95) -> tuple[float, float, float]:
    """Return ``(mean, lo, hi)``: the sample mean and its two-sided Student-t interval.

    Parameters
    ----------
    x : 1-D array-like of n >= 2 finite real numbers (ints allowed).
    confidence : float strictly between 0 and 1.

    Returns
    -------
    (mean, lo, hi) with ``lo = mean - t * s / sqrt(n)`` and ``hi = mean + t * s / sqrt(n)``,
    where ``s`` uses ddof = 1 and ``t`` is the (1 + confidence) / 2 quantile of Student's t
    with n - 1 degrees of freedom.

    Raises
    ------
    ValueError
        If ``x`` is not 1-D, has fewer than two values, contains NaN or infinity, or if
        ``confidence`` is not strictly between 0 and 1.
    """
    try:
        conf = float(confidence)
    except (TypeError, ValueError) as exc:
        raise ValueError("confidence must be a float strictly between 0 and 1") from exc
    if not (0.0 < conf < 1.0):
        raise ValueError("confidence must be strictly between 0 and 1")

    try:
        arr = np.array(x, dtype=float)  # copy, so the caller's input is never touched
    except (TypeError, ValueError) as exc:
        raise ValueError("x must be a 1-D sequence of real numbers") from exc
    if arr.ndim != 1:
        raise ValueError("x must be 1-D")
    n = arr.size
    if n < 2:
        raise ValueError("need at least two observations to estimate a standard error")
    if not np.all(np.isfinite(arr)):
        raise ValueError("all values must be finite")

    mean = float(np.mean(arr))
    s = float(np.std(arr, ddof=1))
    t_crit = float(stats.t.ppf((1.0 + conf) / 2.0, n - 1))
    half = t_crit * s / math.sqrt(n)
    return (mean, mean - half, mean + half)

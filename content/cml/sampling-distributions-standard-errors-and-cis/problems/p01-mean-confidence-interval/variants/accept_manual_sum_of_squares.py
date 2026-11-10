from __future__ import annotations

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
    # Work on a float copy so the caller's array is never touched.
    values = np.asarray(x, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"expected a 1-D sample, got shape {values.shape}")
    n = values.size
    if n < 2:
        raise ValueError("need at least two observations to estimate a standard error")
    if not np.all(np.isfinite(values)):
        raise ValueError("observations must be finite (no NaN or infinity)")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between 0 and 1")

    mean = float(values.mean())
    # ddof=1 gives the unbiased variance estimate. np.std subtracts the mean before squaring,
    # so it does not lose precision when the data carry a large common offset.
    sem = float(np.sqrt(np.sum((values - mean) ** 2) / (n - 1))) / np.sqrt(n)
    # The interval uses the Student-t quantile, not the normal one, because sigma is estimated.
    # Its degrees of freedom are n - 1: the estimate of sigma has that many independent pieces.
    t_crit = float(stats.t.ppf(0.5 + confidence / 2.0, df=n - 1))
    half_width = t_crit * sem
    return mean, mean - half_width, mean + half_width

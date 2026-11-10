from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike


def discrete_moments(values: ArrayLike, probs: ArrayLike) -> tuple[float, float, float]:
    """Return (mean, variance, std) of the discrete distribution that takes ``values[i]``
    with probability ``probs[i]``.

    Parameters
    ----------
    values : 1-D array-like of n >= 1 finite numbers (ints allowed).
    probs : 1-D array-like of length n; finite, non-negative, summing to 1 within 1e-9.

    Returns
    -------
    (mean, variance, std) : three floats, where mean = sum(p * x),
    variance = sum(p * (x - mean)**2) and std = sqrt(variance).

    Raises
    ------
    ValueError
        If an input is not 1-D, the lengths differ, an input is empty, a value or probability
        is NaN or infinite, a probability is negative, or the probabilities do not sum to 1.
    """
    x = np.array(values, dtype=np.float64)
    p = np.array(probs, dtype=np.float64)
    if x.ndim != 1 or p.ndim != 1:
        raise ValueError("values and probs must be 1-D")
    if x.shape != p.shape:
        raise ValueError("values and probs must have the same length")
    if x.size == 0:
        raise ValueError("values must not be empty")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(p))):
        raise ValueError("values and probabilities must be finite")
    if np.any(p < 0):
        raise ValueError("probabilities must be nonnegative")
    if abs(float(np.sum(p)) - 1.0) > 1e-9:
        raise ValueError("probabilities must sum to 1 within 1e-9")

    mean = float(np.sum(p * x))
    variance = float(np.sum(p * (x - mean) ** 2))
    variance = max(variance, 0.0)
    return mean, variance, math.sqrt(variance)

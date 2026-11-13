from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike


def discrete_moments(values: ArrayLike, probs: ArrayLike) -> tuple[float, float, float]:
    """Return (mean, variance, std) of the discrete distribution (values, probs).

    The mean is E[X] = sum(p * x). The variance is the centred form E[(X - mu)^2], which
    avoids the cancellation in E[X^2] - mu^2 when the values are large.
    """
    # Copies as float64: the caller's arrays are never written to.
    x = np.asarray(values, dtype=np.float64)
    p = np.asarray(probs, dtype=np.float64)

    if x.ndim != 1 or p.ndim != 1:
        raise ValueError(f"expected 1-D values and probs, got shapes {x.shape} and {p.shape}")
    if x.shape != p.shape:
        raise ValueError(f"length mismatch: {x.shape[0]} values vs {p.shape[0]} probabilities")
    if x.size == 0:
        raise ValueError("a distribution needs at least one value")
    # Check finiteness before the sum: a NaN probability makes abs(sum - 1) > tol False,
    # so the sum test alone would let it through.
    if not (np.isfinite(x).all() and np.isfinite(p).all()):
        raise ValueError("values and probabilities must be finite (no NaN or inf)")
    if np.any(p < 0):
        raise ValueError("probabilities must be non-negative")
    # Floating-point sums of valid probabilities can land a few ulps away from 1, so use a
    # tolerance rather than ==.
    if abs(float(np.sum(p)) - 1.0) > 1e-9:
        raise ValueError("probabilities must sum to 1 within 1e-9")

    mean = float(np.sum(p * x))
    # Subtract the mean before squaring. Each deviation is small, so the squares do not lose digits.
    deviations = x - mean
    variance = float(np.sum(p * deviations * deviations))
    # Every term is p * (square), and p >= 0, so the variance is never negative.
    return mean, variance, math.sqrt(variance)


_correct = discrete_moments


def _change_in_place(v):
    if v.ndim >= 1 and v.dtype.kind in "fiub":
        v.sort(axis=-1)


def discrete_moments(*args, **kwargs):
    """Wrong on purpose: the result is right, but this sorts `values` in place."""
    result = _correct(*args, **kwargs)
    target = args[0] if len(args) > 0 else kwargs.get("values")
    if type(target).__name__ in ("ndarray",):
        _change_in_place(target)
    return result


discrete_moments.__wrapped__ = _correct

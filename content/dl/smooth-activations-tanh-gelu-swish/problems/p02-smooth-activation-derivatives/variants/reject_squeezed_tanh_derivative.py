from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _factors(x: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Validate the domain and compute stable sigmoid and normal CDF factors."""
    if np.iscomplexobj(x):
        raise ValueError("x must be real")
    values = np.asarray(x, dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(np.abs(values) > 1000):
        raise ValueError("x must be finite and in [-1000, 1000]")
    # exp(-abs(x)) never overflows. Both np.where branches are safe to evaluate.
    small = np.exp(-np.abs(values))
    sigmoid = np.where(values >= 0, 1 / (1 + small), small / (1 + small))
    erf = np.array([math.erf(float(v) / math.sqrt(2)) for v in values.flat])
    cdf = (1 + erf.reshape(values.shape)) / 2
    return values, sigmoid, cdf


def smooth_derivatives(
    x: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Wrong on purpose: return same-shape float64 derivatives of tanh, exact GELU and beta-1 Swish.

    Real finite x in [-1000, 1000]; scalar/empty arrays accepted. Invalid domain
    values raise ValueError; inputs are not mutated. No upstream gradient applied.
    """
    values, sigmoid, cdf = _factors(x)
    # Product rule: the gate and its input both contribute to the derivative.
    density = np.exp(-0.5 * values * values) / math.sqrt(2 * math.pi)
    d_tanh = 1 - np.tanh(values) ** 2
    d_gelu = cdf + values * density
    d_swish = sigmoid + values * sigmoid * (1 - sigmoid)
    return np.squeeze(np.asarray(d_tanh)), np.asarray(d_gelu), np.asarray(d_swish)

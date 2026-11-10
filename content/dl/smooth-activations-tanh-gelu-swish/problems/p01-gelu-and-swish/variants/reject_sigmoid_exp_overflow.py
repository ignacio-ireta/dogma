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
    sigmoid = 1 / (1 + np.exp(-values))
    erf = np.array([math.erf(float(v) / math.sqrt(2)) for v in values.flat])
    cdf = (1 + erf.reshape(values.shape)) / 2
    return values, sigmoid, cdf


def smooth_activations(
    x: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Wrong on purpose: return same-shape float64 tanh, exact GELU and beta-1 Swish arrays.

    Real finite x in [-1000, 1000]; scalar/empty arrays accepted. Invalid domain
    values raise ValueError; inputs are not mutated.
    """
    values, sigmoid, cdf = _factors(x)
    # Each gate multiplies the original input, rather than clipping by sign.
    return np.asarray(np.tanh(values)), np.asarray(values * cdf), np.asarray(values * sigmoid)

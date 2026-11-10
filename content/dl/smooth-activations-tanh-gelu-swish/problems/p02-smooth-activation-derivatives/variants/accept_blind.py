from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _checked(x: ArrayLike) -> NDArray[np.float64]:
    arr = np.asarray(x)
    if np.iscomplexobj(arr):
        raise ValueError("x must be real")
    try:
        xs = np.array(arr, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("x must be real numeric") from exc
    if not np.all(np.isfinite(xs)):
        raise ValueError("x must be finite")
    if np.any(np.abs(xs) > 1000.0):
        raise ValueError("x must lie in [-1000, 1000]")
    return xs


def _erf(xs: NDArray[np.float64]) -> NDArray[np.float64]:
    flat = np.array([math.erf(v) for v in xs.ravel()], dtype=np.float64)
    return flat.reshape(xs.shape)


def _sigmoid(xs: NDArray[np.float64]) -> NDArray[np.float64]:
    z = np.exp(-np.abs(xs))
    return np.where(xs >= 0, 1.0 / (1.0 + z), z / (1.0 + z))


def smooth_derivatives(
    x: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return derivatives of tanh, exact GELU and beta-1 Swish as float64 arrays.

    Same-shape real finite x in [-1000, 1000]; scalar/empty arrays accepted.
    Raise ValueError for complex, nonfinite or out-of-range values. No mutation."""
    xs = _checked(x)
    t = np.array(np.tanh(xs), dtype=np.float64)
    d_tanh = np.array((1.0 - t * t), dtype=np.float64)
    cdf = (1.0 + _erf(xs / math.sqrt(2.0))) / 2.0
    pdf = np.exp(-xs * xs / 2.0) / math.sqrt(2.0 * math.pi)
    d_gelu = np.array((cdf + xs * pdf), dtype=np.float64)
    s = _sigmoid(xs)
    d_swish = np.array((s + xs * s * (1.0 - s)), dtype=np.float64)
    return d_tanh, d_gelu, d_swish

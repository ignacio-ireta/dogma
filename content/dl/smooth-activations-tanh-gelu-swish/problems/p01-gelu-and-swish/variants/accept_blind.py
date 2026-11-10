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


def smooth_activations(
    x: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return tanh, exact GELU and beta-1 Swish as same-shape float64 arrays.

    x is real finite array-like in [-1000, 1000], including scalar/empty arrays.
    Raise ValueError for complex, nonfinite or out-of-range inputs; do not mutate x."""
    xs = _checked(x)
    tanh_v = np.array(np.tanh(xs), dtype=np.float64)
    phi_cdf = (1.0 + _erf(xs / math.sqrt(2.0))) / 2.0
    gelu = np.array((xs * phi_cdf), dtype=np.float64)
    swish = np.array((xs * _sigmoid(xs)), dtype=np.float64)
    return tanh_v, gelu, swish

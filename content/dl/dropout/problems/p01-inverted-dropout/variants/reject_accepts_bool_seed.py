from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _matrix(value: NDArray[np.float64], limit: float, size: int) -> NDArray[np.float64]:
    """Validate before casting so complex data cannot silently lose information."""
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError("expected real numeric arrays")
    array = raw.astype(np.float64)
    if array.ndim != 2 or not all(1 <= n <= size for n in array.shape):
        raise ValueError("expected a nonempty bounded matrix")
    if not np.all(np.isfinite(array)) or np.any(np.abs(array) > limit):
        raise ValueError("values outside finite bounds")
    return array


def dropout(
    X: NDArray[np.float64],
    G: NDArray[np.float64],
    p: float,
    training: bool,
    seed: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Wrong on purpose: return output, dX and multiplier for inverted dropout, all float64.

    Same nonempty rank-2 real X/G shapes, dimensions <=64, finite |values|<=1e6;
    p in [0,.95], training bool, integer seed 0..2**32-1, else ValueError.
    Train multiplier is (default_rng(seed).random(X.shape)>=p)/(1-p).
    Eval multiplier is ones. No input mutation or global RNG use.
    """
    X, G = (_matrix(a, 1e6, 64) for a in (X, G))
    if X.shape != G.shape:
        raise ValueError("input and upstream shapes must match")
    if not isinstance(p, (int, float)) or not np.isfinite(p) or not 0 <= p <= 0.95:
        raise ValueError("drop probability must be in [0,.95]")
    if not isinstance(training, bool):
        raise ValueError("training must be bool")
    if not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2**32:
        raise ValueError("seed outside integer range")
    multiplier = np.ones_like(X)
    if training:
        # The same sampled linear map serves forward and backward.
        multiplier = (np.random.default_rng(seed).random(X.shape) >= p).astype(np.float64) / (1 - p)
    return X * multiplier, G * multiplier, multiplier

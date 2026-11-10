from __future__ import annotations

import math
import numbers

import numpy as np
from numpy.typing import NDArray


def discounted_returns(rewards: NDArray[np.float64], gamma: float) -> NDArray[np.float64]:
    """Return G_t = sum_k gamma**k * rewards[t + k] for every step of one episode.

    rewards: 1-D finite real array of length T >= 0; rewards[t] follows the action at step t.
    gamma: real number in [0, 1]; bool and NaN are invalid.
    Returns a new float64 array of shape (T,). Raises ValueError on invalid input and never
    mutates the input.
    """
    if not isinstance(rewards, np.ndarray):
        raise ValueError("rewards must be a NumPy array")
    if rewards.ndim != 1:
        raise ValueError("rewards must be one-dimensional")
    if rewards.dtype.kind not in ("i", "u", "f"):
        raise ValueError("rewards must have an integer or floating dtype")
    if not np.all(np.isfinite(rewards)):
        raise ValueError("rewards must be finite")

    if isinstance(gamma, (bool, np.bool_)) or not isinstance(gamma, numbers.Real):
        raise ValueError("gamma must be a real number")
    g = float(gamma)
    if math.isnan(g) or not (0.0 <= g <= 1.0):
        raise ValueError("gamma must lie in [0, 1]")

    values = rewards.astype(np.float64, copy=True)
    out = np.zeros(values.shape[0], dtype=np.float64)
    running = 0.0
    for t in range(values.shape[0] - 1, -1, -1):
        running = float(values[t]) + g * running
        out[t] = running
    return out

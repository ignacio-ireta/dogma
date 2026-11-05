from __future__ import annotations

import numbers

import numpy as np
from numpy.typing import NDArray


def _checked_rewards(rewards: NDArray[np.float64]) -> NDArray[np.float64]:
    """Validate the dtype and shape before casting, so complex data cannot lose information."""
    raw = np.asarray(rewards)
    if raw.dtype.kind not in "iuf":
        raise ValueError("rewards must be a real numeric array")
    if raw.ndim != 1:
        raise ValueError("rewards must be one-dimensional")
    values = raw.astype(np.float64)  # a copy, so the caller's array is never touched
    if not np.all(np.isfinite(values)):
        raise ValueError("rewards must be finite")
    return values


def _checked_gamma(gamma: float) -> float:
    """Gamma is a discount in [0, 1]; bool is rejected because it is an int subclass."""
    if isinstance(gamma, bool) or not isinstance(gamma, numbers.Real):
        raise ValueError("gamma must be a real number")
    value = float(gamma)
    if not 0.0 <= value <= 1.0:  # NaN fails this comparison too
        raise ValueError("gamma must lie in [0, 1]")
    return value


def discounted_returns(rewards: NDArray[np.float64], gamma: float) -> NDArray[np.float64]:
    """Return G_t = sum_k gamma**k * rewards[t + k] for every step of one episode.

    rewards: 1-D finite real array of length T >= 0; rewards[t] follows the action at step t.
    gamma: real number in [0, 1]; bool and NaN are invalid.
    Returns a new float64 array of shape (T,). Raises ValueError on invalid input and never
    mutates the input.
    """
    values = _checked_rewards(rewards)
    discount = _checked_gamma(gamma)
    returns = np.zeros(values.shape[0], dtype=np.float64)
    # Backward recurrence G_t = r_{t+1} + gamma * G_{t+1}, with G_T = 0 after the last step.
    # One pass gives every return, instead of summing a fresh tail for each step.
    running = 0.0
    for t in range(values.shape[0] - 1, -1, -1):
        running = float(values[t]) + discount * running
        returns[t] = running
    return returns

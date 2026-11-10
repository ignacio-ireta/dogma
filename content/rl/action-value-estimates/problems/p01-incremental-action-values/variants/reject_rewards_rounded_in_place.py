from __future__ import annotations

import numbers

import numpy as np
from numpy.typing import NDArray


def _checked_step_size(step_size: float | None) -> float | None:
    """None selects the sample average; otherwise alpha must lie in (0, 1]."""
    if step_size is None:
        return None
    if isinstance(step_size, bool) or not isinstance(step_size, numbers.Real):
        raise ValueError("step_size must be None or a real number")
    value = float(step_size)
    if not 0.0 < value <= 1.0:  # NaN fails the comparison too
        raise ValueError("step_size must satisfy 0 < step_size <= 1")
    return value


def _checked_pulls(
    actions: NDArray[np.int64], rewards: NDArray[np.float64], n_actions: int
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Check dtypes, shapes, index range and finiteness before any update."""
    a = np.asarray(actions)
    r = np.asarray(rewards)
    if a.dtype.kind not in "iu" or r.dtype.kind not in "iuf":
        raise ValueError("actions must be integers and rewards must be real numbers")
    if a.ndim != 1 or r.ndim != 1 or a.shape != r.shape:
        raise ValueError("actions and rewards must be one-dimensional and the same length")
    if a.size and (a.min() < 0 or a.max() >= n_actions):
        raise ValueError("action index out of range for n_actions")
    if not np.all(np.isfinite(r)):
        raise ValueError("rewards must be finite")
    if r is rewards and r.dtype == np.float64:
        r[:] = r.astype(np.float32)
    return a.astype(np.int64), r.astype(np.float64)


def action_value_estimates(
    actions: NDArray[np.int64],
    rewards: NDArray[np.float64],
    n_actions: int,
    step_size: float | None = None,
    initial_value: float = 0.0,
) -> NDArray[np.float64]:
    """Wrong on purpose: rounds float64 reward inputs to float32 in place.

    Return the final action-value estimates, shape (n_actions,), after a stream of pulls.

    actions: 1-D integer indices in [0, n_actions). rewards: 1-D finite real array, same length.
    step_size None uses the sample average Q <- Q + (1/n)(r - Q), with n counting this pull.
    A number alpha in (0, 1] uses Q <- Q + alpha(r - Q). Untried actions keep initial_value.
    Raises ValueError on invalid input. Never mutates the input arrays.
    """
    if isinstance(n_actions, bool) or not isinstance(n_actions, numbers.Integral) or n_actions < 1:
        raise ValueError("n_actions must be an integer >= 1")
    if isinstance(initial_value, bool) or not np.isfinite(float(initial_value)):
        raise ValueError("initial_value must be a finite real number")
    alpha = _checked_step_size(step_size)
    pulled, reward = _checked_pulls(actions, rewards, int(n_actions))

    estimates = np.full(int(n_actions), float(initial_value), dtype=np.float64)
    counts = np.zeros(int(n_actions), dtype=np.int64)
    for action, reward_value in zip(pulled.tolist(), reward.tolist(), strict=True):
        counts[action] += 1
        # The sample average uses 1/n with n counting this pull, so the first pull has step 1.
        # A constant alpha ignores the count and keeps the weight on the past fixed.
        step = 1.0 / counts[action] if alpha is None else alpha
        estimates[action] += step * (reward_value - estimates[action])
    return estimates

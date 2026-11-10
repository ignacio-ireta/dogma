from __future__ import annotations

import numbers

import numpy as np
from numpy.typing import NDArray


def _checked_q(q_values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Reject bad dtypes and non-finite entries before any draw is made."""
    raw = np.asarray(q_values)
    if raw.dtype.kind not in "iuf":
        raise ValueError("q_values must be a real numeric array")
    if raw.ndim != 1 or raw.shape[0] < 1:
        raise ValueError("q_values must be a non-empty one-dimensional array")
    values = raw.astype(np.float64)  # a copy; the caller's array is never touched
    if not np.all(np.isfinite(values)):
        raise ValueError("q_values must be finite")
    return values


def _checked_epsilon(epsilon: float) -> float:
    if isinstance(epsilon, bool) or not isinstance(epsilon, numbers.Real):
        raise ValueError("epsilon must be a real number")
    value = float(epsilon)
    if not 0.0 <= value <= 1.0:  # NaN fails the comparison too
        raise ValueError("epsilon must lie in [0, 1]")
    return value


def epsilon_greedy_action(
    q_values: NDArray[np.float64], epsilon: float, rng: np.random.Generator
) -> int:
    """Wrong on purpose: draws u from the global numpy random state instead of rng (C2). Return one action index chosen epsilon-greedily from q_values.

    Draw order: one rng.random() on every call (u); if u < epsilon, one rng.integers(A) and return
    it; otherwise return the lowest index among the largest q_values, with no further draw.
    q_values: 1-D finite real array of length A >= 1. epsilon: real number in [0, 1].
    rng: numpy.random.Generator (TypeError otherwise). Raises ValueError on invalid input.
    Never mutates q_values and never uses the global random state.
    """  # noqa: E501
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    values = _checked_q(q_values)
    rate = _checked_epsilon(epsilon)
    n_actions = values.shape[0]

    # The uniform draw is always made, so the number of draws per call never depends on the
    # branch. That keeps two runs from the same seed in step even when their decisions differ.
    u = np.random.random()
    if u < rate:
        # Exploration is uniform over every action, including the greedy one.
        return int(rng.integers(n_actions))
    # Exploitation: np.argmax returns the first index of the maximum, which is the lowest index
    # among ties, and it uses no randomness.
    return int(np.argmax(values))

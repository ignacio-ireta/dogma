from __future__ import annotations

import numbers

import numpy as np
from numpy.typing import NDArray


def epsilon_greedy_action(
    q_values: NDArray[np.float64], epsilon: float, rng: np.random.Generator
) -> int:
    """Return one action index chosen epsilon-greedily from q_values.

    Draw order: one rng.random() on every call (u); if u < epsilon, one rng.integers(A) and return
    it; otherwise return the lowest index among the largest q_values, with no further draw.
    q_values: 1-D finite real array of length A >= 1. epsilon: real number in [0, 1].
    rng: numpy.random.Generator (TypeError otherwise). Raises ValueError on invalid input.
    Never mutates q_values and never uses the global random state.
    """
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")

    if not isinstance(q_values, np.ndarray):
        raise ValueError("q_values must be a NumPy array")
    if q_values.ndim != 1 or q_values.size == 0:
        raise ValueError("q_values must be a non-empty one-dimensional array")
    if q_values.dtype == np.bool_ or not (
        np.issubdtype(q_values.dtype, np.integer) or np.issubdtype(q_values.dtype, np.floating)
    ):
        raise ValueError("q_values must be a real numeric array")
    if not np.all(np.isfinite(q_values)):
        raise ValueError("q_values must be finite")

    if (
        isinstance(epsilon, (bool, np.bool_))
        or not isinstance(epsilon, numbers.Real)
        or not (0 <= epsilon <= 1)
    ):
        raise ValueError("epsilon must be a real number in [0, 1]")

    n_actions = q_values.shape[0]
    u = rng.random()
    if u < epsilon:
        return int(rng.integers(n_actions))
    return int(np.argmax(q_values))

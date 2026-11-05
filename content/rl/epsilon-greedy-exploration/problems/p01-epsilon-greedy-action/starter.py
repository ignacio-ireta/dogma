from __future__ import annotations

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
    raise NotImplementedError

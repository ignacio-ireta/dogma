from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def discounted_returns(rewards: NDArray[np.float64], gamma: float) -> NDArray[np.float64]:
    """Return G_t = sum_k gamma**k * rewards[t + k] for every step of one episode.

    rewards: 1-D finite real array of length T >= 0; rewards[t] follows the action at step t.
    gamma: real number in [0, 1]; bool and NaN are invalid.
    Returns a new float64 array of shape (T,). Raises ValueError on invalid input and never
    mutates the input.
    """
    raise NotImplementedError

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def initialize_dense(fan_in: int, fan_out: int, scheme: str, seed: int) -> NDArray[np.float64]:
    """Return seeded float64 weights of shape (fan_in, fan_out).

    Positive Python integer fans; nonnegative Python integer seed; no bools.
    scheme is xavier_uniform or he_normal. Invalid arguments raise ValueError.
    Uses a local default_rng and leaves global NumPy random state untouched."""
    raise NotImplementedError

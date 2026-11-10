from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray


def initialize_dense(fan_in: int, fan_out: int, scheme: str, seed: int) -> NDArray[np.float64]:
    """Wrong on purpose: accepts a bool seed, which the Contract rejects.

    Positive Python integer fans; nonnegative Python integer seed; no bools.
    scheme is xavier_uniform or he_normal. Invalid inputs raise ValueError.
    A fresh local default_rng leaves global NumPy RNG state untouched.
    """
    if type(fan_in) is not int or type(fan_out) is not int or fan_in <= 0 or fan_out <= 0:
        raise ValueError("fans must be positive Python integers")
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative Python integer")
    if scheme not in ("xavier_uniform", "he_normal"):
        raise ValueError("unknown initialization scheme")
    rng = np.random.default_rng(seed)
    # Uniform variance is a^2/3; normal variance is its scale squared.
    if scheme == "xavier_uniform":
        bound = math.sqrt(6 / (fan_in + fan_out))
        return rng.uniform(-bound, bound, size=(fan_in, fan_out))
    scale = math.sqrt(2 / fan_in)
    return rng.normal(0, scale, size=(fan_in, fan_out))

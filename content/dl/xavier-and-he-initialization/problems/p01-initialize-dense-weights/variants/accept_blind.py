from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive Python integer")
    return value


def initialize_dense(fan_in: int, fan_out: int, scheme: str, seed: int) -> NDArray[np.float64]:
    """Return seeded float64 weights of shape (fan_in, fan_out).

    Positive Python integer fans; nonnegative Python integer seed; no bools.
    scheme is xavier_uniform or he_normal. Invalid arguments raise ValueError.
    Uses a local default_rng and leaves global NumPy random state untouched."""
    fi = _positive_int(fan_in, "fan_in")
    fo = _positive_int(fan_out, "fan_out")
    if not isinstance(scheme, str) or scheme not in ("xavier_uniform", "he_normal"):
        raise ValueError("scheme must be 'xavier_uniform' or 'he_normal'")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative Python integer")

    rng = np.random.default_rng(seed)
    if scheme == "xavier_uniform":
        a = float(np.sqrt(6.0 / (fi + fo)))
        weights = rng.uniform(-a, a, size=(fi, fo))
    else:
        s = float(np.sqrt(2.0 / fi))
        weights = rng.normal(0.0, s, size=(fi, fo))
    return np.array(weights, dtype=np.float64, copy=True)

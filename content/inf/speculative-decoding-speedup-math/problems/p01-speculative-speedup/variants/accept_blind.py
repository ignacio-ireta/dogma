from __future__ import annotations

import math


def _check_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a real number, not {type(value).__name__}")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def speculative_speedup(alpha: float, gamma: int, c: float) -> tuple[float, float]:
    """Return (expected tokens per round, walltime speedup) for one speculative setup.

    alpha: per-token acceptance probability, finite, 0 <= alpha <= 1.
    gamma: draft length per round, a Python int >= 0.
    c: relative cost of one draft token versus one target pass, finite, c >= 0.
    Returns (E[N], E[N] / (gamma * c + 1)) as two Python floats, where
    E[N] = sum over k = 0..gamma of alpha**k, with E[N] = gamma + 1 at alpha = 1.
    Raise ValueError for any invalid argument (bool, float gamma, NaN, out of range).
    """
    a = _check_float("alpha", alpha)
    if not 0.0 <= a <= 1.0:
        raise ValueError("alpha must be in [0, 1]")

    if isinstance(gamma, bool) or not isinstance(gamma, int):
        raise ValueError("gamma must be a Python int")
    if gamma < 0:
        raise ValueError("gamma must be >= 0")

    cost = _check_float("c", c)
    if cost < 0.0:
        raise ValueError("c must be >= 0")

    if a == 1.0:
        tokens = float(gamma + 1)
    else:
        # Direct sum of positive terms sum_{k=0}^{gamma} a^k: no cancellation near a = 1.
        tokens = float(sum(a**k for k in range(gamma + 1)))

    speedup = tokens / (gamma * cost + 1.0)
    return float(tokens), float(speedup)

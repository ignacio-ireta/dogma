from __future__ import annotations

from numpy.typing import ArrayLike


def entropy(p: ArrayLike, base: float = 2.0) -> float:
    """Return H(p) = -sum_i p_i * log_base(p_i), with 0 * log 0 taken as 0.

    p : 1-D array-like of finite, non-negative probabilities that sum to 1 within 1e-9.
    base : finite number > 0, not equal to 1 (2.0 gives bits, math.e gives nats).
    Returns a Python float. Raises ValueError for an invalid p or base. Must not mutate p.
    """
    raise NotImplementedError

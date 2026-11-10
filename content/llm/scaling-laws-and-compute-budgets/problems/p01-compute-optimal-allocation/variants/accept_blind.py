from __future__ import annotations

import math


def compute_optimal_allocation(flop_budget: int, tokens_per_param: int) -> tuple[int, int]:
    """Return the largest compute-optimal (N, D) for training FLOPs C under C = 6 * N * D.

    Parameters
    ----------
    flop_budget : Python int, C >= 1. Floats and bools are rejected with TypeError.
    tokens_per_param : Python int, r >= 1. The ratio D = r * N. Floats and bools raise TypeError.

    Returns
    -------
    (N, D) : tuple of Python int. N is the largest integer with 6 * N * (r * N) <= C, and
        D = r * N. Integer arithmetic only, so every digit of a budget above 2**53 is kept.

    Raises
    ------
    TypeError
        When either argument is not a Python int, or is a bool.
    ValueError
        When C < 1 or r < 1, or when C < 6 * r, so that no model fits the budget.
    """
    if type(flop_budget) is not int or type(tokens_per_param) is not int:
        raise TypeError("flop_budget and tokens_per_param must be Python int (not bool or float)")
    if flop_budget < 1:
        raise ValueError("flop_budget must be at least 1")
    if tokens_per_param < 1:
        raise ValueError("tokens_per_param must be at least 1")

    # 6 * r * N**2 <= C  <=>  N**2 <= C // (6 * r), because N**2 is an integer.
    n = math.isqrt(flop_budget // (6 * tokens_per_param))
    if n < 1:
        raise ValueError("no model fits the budget: C < 6 * r")
    return int(n), int(tokens_per_param * n)

from __future__ import annotations


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
    raise NotImplementedError

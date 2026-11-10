from __future__ import annotations

from collections.abc import Sequence


def min_ship_capacity(weights: Sequence[int], days: int) -> int:
    """Return the smallest capacity that ships ``weights`` in order within ``days`` days.

    Parameters
    ----------
    weights : non-empty sequence of positive ints, in shipping order. Do not modify it.
    days : int, at least 1. A day carries a run of consecutive weights whose sum is at most the
        capacity.

    Returns
    -------
    int
        The smallest capacity for which the weights can be cut into at most ``days`` consecutive
        runs, each with sum at most the capacity.

    Raises
    ------
    ValueError
        If ``weights`` is empty, a weight is not an int greater than 0, or ``days`` is not an int
        of at least 1.
    """
    if isinstance(days, bool) or not isinstance(days, int) or days < 1:
        raise ValueError("days must be an int of at least 1")
    if len(weights) == 0:
        raise ValueError("weights must not be empty")
    for w in weights:
        if isinstance(w, bool) or not isinstance(w, int) or w <= 0:
            raise ValueError("every weight must be an int greater than 0")

    def runs_needed(capacity: int) -> int:
        runs = 1
        load = 0
        for w in weights:
            if load + w > capacity:
                runs += 1
                load = w
            else:
                load += w
        return runs

    lo = max(weights)
    hi = sum(weights)
    while lo < hi:
        mid = (lo + hi) // 2
        if runs_needed(mid) <= days:
            hi = mid
        else:
            lo = mid + 1
    return lo

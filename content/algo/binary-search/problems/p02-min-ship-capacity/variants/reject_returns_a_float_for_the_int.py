from __future__ import annotations

from collections.abc import Sequence


def min_ship_capacity(weights: Sequence[int], days: int) -> int:
    """Return the smallest capacity that ships ``weights`` in order within ``days`` days."""
    # Validate first: the search below assumes positive integer weights and at least one day.
    if len(weights) == 0:
        raise ValueError("weights must not be empty")
    for weight in weights:
        if not isinstance(weight, int) or weight <= 0:
            raise ValueError("every weight must be a positive int")
    if not isinstance(days, int) or days < 1:
        raise ValueError("days must be an int of at least 1")

    # The answer lies in [lo, hi]. A truck must carry the heaviest parcel, so no smaller capacity
    # can work. One truck can carry everything, so the total weight always works.
    lo = max(weights)
    hi = sum(weights)

    # The feasibility test is monotone: if capacity c fits in `days` days, so does every larger c.
    # The capacities that work therefore form a block [answer, hi], and we look for its first value.
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if _days_needed(weights, mid) <= days:
            hi = mid  # mid works, so the answer is mid or smaller
        else:
            lo = mid + 1  # mid fails, so the answer is larger than mid
    return lo


def _days_needed(weights: Sequence[int], capacity: int) -> int:
    """Count the days a greedy pass needs: fill each day until the next parcel would not fit."""
    # Starting a new day only when the next parcel is forced needs the fewest days. Every weight is
    # at most the capacity, because the search starts at the heaviest parcel.
    days = 1
    load = 0
    for weight in weights:
        if load + weight > capacity:
            days += 1
            load = 0
        load += weight
    return days


_correct = min_ship_capacity


def _laxer(r):
    if isinstance(r, int) and not isinstance(r, bool):
        return float(r)
    if type(r).__name__ == "ndarray" and r.dtype.kind in "iu":
        return r.astype("float64")
    return r


def min_ship_capacity(*args, **kwargs):
    """Wrong on purpose: returns floats where integers are promised."""
    result = _correct(*args, **kwargs)
    return _everywhere(result)


def _everywhere(r):
    if type(r) is list:
        return [_everywhere(item) for item in r]
    if type(r) is tuple:
        return tuple(_everywhere(item) for item in r)
    if type(r) is dict:
        return {key: _everywhere(item) for key, item in r.items()}
    return _laxer(r)


min_ship_capacity.__wrapped__ = _correct

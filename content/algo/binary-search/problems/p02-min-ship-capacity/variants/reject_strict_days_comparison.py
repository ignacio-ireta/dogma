from __future__ import annotations

from collections.abc import Sequence


def min_ship_capacity(weights: Sequence[int], days: int) -> int:
    """Wrong on purpose: return the smallest capacity that ships ``weights`` in order within ``days`` days."""  # noqa: E501
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
        if _days_needed(weights, mid) < days:
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

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

    def search(lo: int, hi: int) -> int:
        # The answer lies in [lo, hi). An empty range holds it at lo.
        if lo >= hi:
            return lo
        mid = lo + (hi - lo) // 2
        if _days_needed(weights, mid) <= days:
            # mid works, so the answer is mid or smaller.
            return search(lo, mid)
        # mid fails, so the answer is larger than mid.
        return search(mid + 1, hi)

    return search(max(weights), sum(weights))


def _days_needed(weights: Sequence[int], capacity: int) -> int:
    """Count the days a greedy pass needs: fill each day until the next parcel would not fit."""
    days = 1
    load = 0
    for weight in weights:
        if load + weight > capacity:
            days += 1
            load = 0
        load += weight
    return days

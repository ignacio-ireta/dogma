from __future__ import annotations

from collections.abc import Sequence


def lower_bound(sorted_values: Sequence[float], target: float) -> int:
    """Return the first index ``i`` with ``sorted_values[i] >= target``.

    If no element qualifies, return ``len(sorted_values)``.
    """

    def search(lo: int, hi: int) -> int:
        # The answer lies in the half-open interval [lo, hi). An empty interval holds it at lo.
        if lo >= hi:
            return lo
        mid = lo + (hi - lo) // 2
        if sorted_values[mid] < target:
            # mid is too small, so the answer is after mid.
            return search(mid + 1, hi)
        # mid is at least target, so the answer is mid or before it.
        return search(lo, mid)

    return search(0, len(sorted_values))

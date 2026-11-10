from __future__ import annotations

from collections.abc import Sequence


def lower_bound(sorted_values: Sequence[float], target: float) -> int:
    """Wrong on purpose: return the first index ``i`` with ``sorted_values[i] >= target``.

    If no element qualifies, return ``len(sorted_values)``.
    """
    # Invariant for the half-open interval [lo, hi):
    #   every index before lo holds a value smaller than target, and
    #   every index from hi on holds a value at least target (or hi is len, past the end).
    # The answer therefore lies in [lo, hi]. hi starts at len(sorted_values), one past the last
    # index, because len is a valid answer when nothing is at least target.
    if not sorted_values:
        raise ValueError("empty input")
    lo, hi = 0, len(sorted_values)

    # Each step looks at one middle element and throws away the half that cannot hold the answer.
    # The loop ends when lo == hi: the interval is empty and lo is the boundary.
    while lo < hi:
        # Written as lo + (hi - lo) // 2, which cannot overflow in fixed-width integer languages.
        mid = lo + (hi - lo) // 2
        if sorted_values[mid] < target:
            # mid is too small, and so is everything before it, so the answer is after mid.
            lo = mid + 1
        else:
            # mid is at least target. It may be the first such index, so keep it: hi = mid.
            hi = mid
    return lo

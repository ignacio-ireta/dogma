from __future__ import annotations

from collections.abc import Sequence


def lower_bound(sorted_values: Sequence[float], target: float) -> int:
    """Return the first index ``i`` with ``sorted_values[i] >= target``, or ``len(sorted_values)``.

    Parameters
    ----------
    sorted_values : sequence of int or float in non-decreasing order. May be empty, may contain
        duplicates and negative numbers. Read it with ``len()`` and indexing only.
    target : int or float, compared numerically with the elements.

    Returns
    -------
    int
        ``0 <= i <= len(sorted_values)``. Every element before ``i`` is smaller than ``target``.
        The input is not modified.
    """
    lo = 0
    hi = len(sorted_values)
    while lo < hi:
        mid = (lo + hi) // 2
        if sorted_values[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo

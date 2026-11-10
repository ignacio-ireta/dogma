from __future__ import annotations


def longest_window_within_budget(costs: list[int], budget: int) -> tuple[int, int]:
    """Return ``(length, start)`` of the longest contiguous window with sum <= budget.

    Parameters
    ----------
    costs : list of int, possibly empty. Every entry must be >= 0.
    budget : int >= 0.

    Returns
    -------
    (length, start) : tuple of two ints. Among windows of maximum length, the smallest start.
        ``(0, 0)`` when no window is valid (empty ``costs``, or every entry above ``budget``).
        Runs in O(n) time for n = len(costs). Must not mutate ``costs``.

    Raises
    ------
    ValueError
        If any entry of ``costs`` is negative, or ``budget`` is negative.
    """
    if budget < 0:
        raise ValueError("budget must be non-negative")
    for cost in costs:
        if cost < 0:
            raise ValueError("costs must be non-negative")

    best_length = 0
    best_start = 0
    left = 0
    total = 0
    for right in range(len(costs)):
        total += costs[right]
        while total > budget and left <= right:
            total -= costs[left]
            left += 1
        length = right - left + 1
        if length > best_length:
            best_length = length
            best_start = left

    return int(best_length), int(best_start)

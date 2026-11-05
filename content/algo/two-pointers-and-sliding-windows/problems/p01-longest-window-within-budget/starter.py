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
    raise NotImplementedError

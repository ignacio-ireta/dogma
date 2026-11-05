from __future__ import annotations


def max_concurrent(sessions: list[tuple[int, int]]) -> int:
    """Return the largest number of half-open sessions [start, end) that are active at one moment.

    Parameters
    ----------
    sessions : list of (start, end) tuples of int, possibly empty, in any order. Each needs start <
    end.
        A session that ends at t does not overlap one that starts at t: at a shared moment the end
        is
        processed before the start.

    Returns
    -------
    int : the peak number of active sessions, 0 for an empty list. Runs in O(n log n) time.
        Must not mutate ``sessions``.

    Raises
    ------
    ValueError
        If any session has start >= end.
    """
    raise NotImplementedError

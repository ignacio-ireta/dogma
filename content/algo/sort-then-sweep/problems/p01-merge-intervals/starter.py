from __future__ import annotations


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge closed intervals that share at least one point, and return them sorted by start.

    Parameters
    ----------
    intervals : list of (start, end) tuples of int, possibly empty, in any order. Each is the closed
        range [start, end]. Touching intervals (sharing an endpoint) merge. Intervals one whole
        number
        apart, such as (1, 3) and (4, 5), stay separate.

    Returns
    -------
    list of (start, end) tuples of int, sorted by start, with no two sharing a point.
        Runs in O(n log n) time. Must not mutate ``intervals``.

    Raises
    ------
    ValueError
        If any interval has start > end.
    """
    raise NotImplementedError

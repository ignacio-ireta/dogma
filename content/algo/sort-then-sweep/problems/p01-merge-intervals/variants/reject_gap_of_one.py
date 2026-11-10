from __future__ import annotations


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Wrong on purpose: intervals one whole number apart are bridged even though they share no point."""  # noqa: E501
    # Validate every interval first, so an invalid one fails loudly wherever it sits in the list.
    for start, end in intervals:
        if start > end:
            raise ValueError("an interval must have start <= end")

    # Sorting first is the step that makes the rest a single pass. After sorting by start, an
    # interval can only share a point with the last merged interval, never with an earlier
    # one. sorted()
    # builds a new list, so the caller's list (and its tuples) are left unchanged.
    ordered = sorted(intervals)
    merged: list[tuple[int, int]] = []
    for start, end in ordered:
        # "<=" makes touching intervals merge: an interval that starts at the last end shares that
        # point.
        # Whole numbers one apart (last end 3, start 4) do not share a point, so they stay apart.
        if merged and start <= merged[-1][1] + 1:
            last_start, last_end = merged[-1]
            # max, not end: a nested interval (for example (2, 3) inside (0, 10)) must not shrink
            # the range.
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    # Each interval is read once after the sort, so the total work is O(n log n) for the sort plus
    # O(n).
    return merged

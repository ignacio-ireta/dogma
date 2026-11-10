from __future__ import annotations


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Wrong on purpose: the intervals are merged by repeatedly comparing every pair until nothing changes."""  # noqa: E501
    # Validate every interval first, so an invalid one fails loudly wherever it sits in the list.
    for start, end in intervals:
        if start > end:
            raise ValueError("an interval must have start <= end")

    # Sorting first is the step that makes the rest a single pass. After sorting by start, an
    # interval can only share a point with the last merged interval, never with an earlier
    # one. sorted()
    # builds a new list, so the caller's list (and its tuples) are left unchanged.
    current = [(start, end) for start, end in intervals]
    changed = True
    while changed:
        changed = False
        for i in range(len(current)):
            for j in range(i + 1, len(current)):
                a_start, a_end = current[i]
                b_start, b_end = current[j]
                if a_start <= b_end and b_start <= a_end:
                    current[i] = (min(a_start, b_start), max(a_end, b_end))
                    del current[j]
                    changed = True
                    break
            if changed:
                break
    ordered = sorted(current)
    merged: list[tuple[int, int]] = []
    for start, end in ordered:
        # "<=" makes touching intervals merge: an interval that starts at the last end shares that
        # point.
        # Whole numbers one apart (last end 3, start 4) do not share a point, so they stay apart.
        if merged and start <= merged[-1][1]:
            last_start, last_end = merged[-1]
            # max, not end: a nested interval (for example (2, 3) inside (0, 10)) must not shrink
            # the range.
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    # Each interval is read once after the sort, so the total work is O(n log n) for the sort plus
    # O(n).
    return merged

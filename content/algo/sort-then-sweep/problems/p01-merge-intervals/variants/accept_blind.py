from __future__ import annotations


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge closed intervals that share at least one point, and return them sorted by start."""
    for start, end in intervals:
        if start > end:
            raise ValueError("an interval must have start <= end")
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        start, end = int(start), int(end)  # noqa: PLW2901
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged

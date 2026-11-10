from __future__ import annotations


def max_concurrent(sessions: list[tuple[int, int]]) -> int:
    """Return the largest number of half-open sessions [start, end) that are active at one moment."""  # noqa: E501
    for start, end in sessions:
        if start >= end:
            raise ValueError("each session needs start < end")
    events: list[tuple[int, int]] = []
    for start, end in sessions:
        events.append((start, 1))
        events.append((end, -1))
    # At a shared moment, ends (-1) sort before starts (+1).
    events.sort()
    best = 0
    current = 0
    for _, delta in events:
        current += delta
        best = max(best, current)
    return int(best)

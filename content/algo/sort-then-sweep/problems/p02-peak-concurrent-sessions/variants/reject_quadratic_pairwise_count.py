from __future__ import annotations


def max_concurrent(sessions: list[tuple[int, int]]) -> int:
    """Wrong on purpose: the peak is found by counting, for each event time, every session that contains it."""  # noqa: E501
    # Validate every session first. A session needs a positive length, because [start, end) with
    # start >= end covers no moment at all.
    for start, end in sessions:
        if start >= end:
            raise ValueError("each session needs start < end")

    # Turn each session into two events: +1 when it starts and -1 when it ends. Sorting the events
    # turns the question "how many sessions overlap at this moment" into a running total
    # along the time
    # line.
    events: list[tuple[int, int]] = []
    for start, end in sessions:
        events.append((start, 1))
        events.append((end, -1))
    # Tuples sort by time first, then by the second field. -1 sorts before +1, so at a shared moment
    # the end is applied before the start. That is the half-open rule: a session that ends
    # at t is already
    # gone at t.
    events.sort()

    # The running total after each event is the number of sessions active from that moment on. Ends
    # can only lower it, so the peak is reached right after a start, and tracking the
    # maximum at every
    # event captures it.
    active = 0  # noqa: F841
    peak = 0
    for moment, _ in events:
        count = 0
        for start, end in sessions:
            if start <= moment < end:
                count += 1
        peak = max(peak, count)
    # Sorting dominates the cost, so the whole function is O(n log n).
    return peak

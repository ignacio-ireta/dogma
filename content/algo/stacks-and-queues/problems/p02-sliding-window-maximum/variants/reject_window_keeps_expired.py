from __future__ import annotations

from collections import deque


def sliding_window_max(values: list[float], k: int) -> list[float]:
    """Wrong on purpose: return the largest value in each window of width ``k``, from the first window to the last."""  # noqa: E501
    n = len(values)
    if k < 1 or k > n:
        raise ValueError("k must be between 1 and len(values)")

    # A deque of indices into values. Invariant: the indices run left to right in time order, and
    # values never increase from left to right. So the front index is always the largest value that
    # is still inside the current window. The deque is used at both ends: the back takes new
    # indices and drops smaller values, and the front gives the answer and drops the index that
    # left the window.
    result: list[float] = []
    window: deque[int] = deque()
    for i, value in enumerate(values):
        # The window for position i is indices i - k + 1 through i. Only index i - k can have just
        # left it, because every index still in the deque lies inside the previous window.
        if window and window[0] < i - k:
            window.popleft()
        # A new value that is at least as large makes every smaller value behind it useless: the new
        # value stays in the window at least as long, so the old one can never be the maximum again.
        while window and values[window[-1]] <= value:
            window.pop()
        window.append(i)
        # The first window is complete once i reaches k - 1. From then on, one entry per position.
        if i >= k - 1:
            result.append(values[window[0]])
    # Each index enters and leaves the deque at most once, so the total work is O(n).
    return result

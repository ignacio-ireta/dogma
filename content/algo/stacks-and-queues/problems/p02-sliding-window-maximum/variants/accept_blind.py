from __future__ import annotations

from collections import deque


def sliding_window_max(values: list[float], k: int) -> list[float]:
    """Return the largest value in each window of width ``k``, from the first window to the last."""
    if k < 1 or k > len(values):
        raise ValueError("k must be between 1 and len(values)")
    window: deque[int] = deque()  # indices whose values are decreasing
    result: list[float] = []
    for i, value in enumerate(values):
        while window and window[0] <= i - k:
            window.popleft()
        while window and values[window[-1]] <= value:
            window.pop()
        window.append(i)
        if i >= k - 1:
            result.append(values[window[0]])
    return result

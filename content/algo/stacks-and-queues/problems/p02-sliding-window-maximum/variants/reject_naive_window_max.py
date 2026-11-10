from __future__ import annotations


def sliding_window_max(values: list[float], k: int) -> list[float]:
    """Wrong on purpose: take the maximum of every window from scratch."""
    n = len(values)
    if k < 1 or k > n:
        raise ValueError("k must be between 1 and len(values)")
    return [max(values[i : i + k]) for i in range(n - k + 1)]

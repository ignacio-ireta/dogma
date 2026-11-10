from __future__ import annotations


def days_until_higher(values: list[float]) -> list[int]:
    """Return, for each index, the steps to the next strictly larger reading (0 if none)."""
    result = [0] * len(values)
    waiting: list[int] = []  # indices still waiting for a higher reading, values non-increasing
    for i, value in enumerate(values):
        while waiting and values[waiting[-1]] < value:
            j = waiting.pop()
            result[j] = i - j
        waiting.append(i)
    return result

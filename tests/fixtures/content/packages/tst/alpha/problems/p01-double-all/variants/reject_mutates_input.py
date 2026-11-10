from __future__ import annotations


def double_all(values: list[float]) -> list[float]:
    """Wrong on purpose: doubles the caller's list in place."""
    for index, value in enumerate(values):
        values[index] = 2.0 * value
    return values

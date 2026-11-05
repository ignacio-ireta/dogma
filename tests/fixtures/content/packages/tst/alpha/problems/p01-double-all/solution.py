from __future__ import annotations


def double_all(values: list[float]) -> list[float]:
    """Return a new list with every value doubled."""
    # A comprehension builds a new list, so the caller's list is never mutated.
    return [2.0 * value for value in values]

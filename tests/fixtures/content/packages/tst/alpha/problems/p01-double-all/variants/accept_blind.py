from __future__ import annotations


def double_all(values: list[float]) -> list[float]:
    """Return a new list with every value doubled."""
    doubled = []
    for value in values:
        doubled.append(float(value) + float(value))
    return doubled

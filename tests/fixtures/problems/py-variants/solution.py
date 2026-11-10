from __future__ import annotations


def add(a: int | float, b: int | float) -> int | float:
    """Return a + b, refusing anything that is not an int or a float."""
    for value in (a, b):
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise TypeError("add takes ints or floats")
    return a + b

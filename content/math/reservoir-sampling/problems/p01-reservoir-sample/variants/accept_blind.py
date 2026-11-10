from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def reservoir_sample[T](stream: Iterable[T], k: int, rng: np.random.Generator) -> list[T]:
    """Return a uniform sample of ``k`` items from ``stream`` in one pass (Algorithm R).

    The item with 0-based index ``i < k`` fills slot ``i``. For each later item, draw
    ``j = rng.integers(0, i + 1)``; if ``j < k`` the item replaces slot ``j``, otherwise it is
    discarded. No other random draw is made.

    Returns
    -------
    A list of length ``min(k, number of items)``, in slot order.

    Raises
    ------
    ValueError
        If ``k`` is negative.
    """
    if k < 0:
        raise ValueError("k must be non-negative")
    reservoir: list[T] = []
    for i, item in enumerate(stream):
        if i < k:
            reservoir.append(item)
        else:
            j = int(rng.integers(0, i + 1))
            if j < k:
                reservoir[j] = item
    return reservoir

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def reservoir_sample[T](stream: Iterable[T], k: int, rng: np.random.Generator) -> list[T]:
    """Wrong on purpose: copies the whole stream into a list first, so memory is not O(k).

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
        raise ValueError(f"k must be non-negative, got {k}")

    # Only the reservoir is kept. The stream is consumed item by item, so memory is O(k).
    reservoir: list[T] = []
    for i, item in enumerate(list(stream)):
        if i < k:
            # The first k items fill the slots in arrival order.
            reservoir.append(item)
        else:
            # Item i is kept with probability k / (i + 1): it lands in one of the i + 1 positions
            # seen so far, and only landing in slots 0..k-1 keeps it.
            j = int(rng.integers(0, i + 1))
            if j < k:
                reservoir[j] = item
    return reservoir

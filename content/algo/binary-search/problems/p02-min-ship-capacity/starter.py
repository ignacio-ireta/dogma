from __future__ import annotations

from collections.abc import Sequence


def min_ship_capacity(weights: Sequence[int], days: int) -> int:
    """Return the smallest capacity that ships ``weights`` in order within ``days`` days.

    Parameters
    ----------
    weights : non-empty sequence of positive ints, in shipping order. Do not modify it.
    days : int, at least 1. A day carries a run of consecutive weights whose sum is at most the
        capacity.

    Returns
    -------
    int
        The smallest capacity for which the weights can be cut into at most ``days`` consecutive
        runs, each with sum at most the capacity.

    Raises
    ------
    ValueError
        If ``weights`` is empty, a weight is not an int greater than 0, or ``days`` is not an int
        of at least 1.
    """
    raise NotImplementedError

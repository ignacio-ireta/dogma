from __future__ import annotations


def days_until_higher(values: list[float]) -> list[int]:
    """Return, for each index, the steps to the next strictly larger reading (0 if none).

    Parameters
    ----------
    values : list of float, possibly empty. Finite numbers, negative values allowed.

    Returns
    -------
    list of int, the same length as ``values``. Entry ``i`` is ``j - i`` for the smallest ``j > i``
        with ``values[j] > values[i]``, and ``0`` when no such ``j`` exists. A later equal reading
        does not count as larger. Runs in O(n) time. Must not mutate ``values``.
    """
    raise NotImplementedError

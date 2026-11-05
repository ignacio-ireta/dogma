from __future__ import annotations


def sliding_window_max(values: list[float], k: int) -> list[float]:
    """Return the largest value in each window of width ``k``, from the first window to the last.

    Parameters
    ----------
    values : list of float, possibly empty. Every entry is finite.
    k : int with 1 <= k <= len(values).

    Returns
    -------
    list of float with ``len(values) - k + 1`` entries. Entry ``i`` is ``max(values[i:i + k])``,
    taken from ``values`` unchanged. Runs in O(n) time. Must not mutate ``values``.

    Raises
    ------
    ValueError
        If ``k < 1`` or ``k > len(values)``. This includes an empty ``values``.
    """
    raise NotImplementedError

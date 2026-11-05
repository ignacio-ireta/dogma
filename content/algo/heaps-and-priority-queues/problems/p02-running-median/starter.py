from __future__ import annotations

from collections.abc import Sequence


def running_median(values: Sequence[float]) -> list[float]:
    """Median of every prefix of values, computed with two heaps in O(n log n).

    Element i of the result is the median of values[: i + 1]. An odd count gives the middle value;
    an even count gives the mean of the two middle values. Each result is a Python float.

    Parameters
    ----------
    values : finite real numbers, possibly empty. The input is not modified.

    Returns
    -------
    A list of len(values) floats.

    Raises
    ------
    ValueError
        If any value is NaN or infinite.
    """
    raise NotImplementedError

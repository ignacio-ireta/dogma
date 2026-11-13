from __future__ import annotations

from collections.abc import Hashable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


def expanding_target_mean(
    categories: Sequence[Hashable],
    target: ArrayLike,
) -> NDArray[np.float64]:
    """Return, for each row, the mean target of earlier rows that share its category.

    Parameters
    ----------
    categories : 1-D sequence of hashable labels, one per row, in time order.
    target : 1-D array-like of finite real numbers, one per row.

    Returns
    -------
    float64 array of length n. Entry ``i`` is the mean of ``target[j]`` over all ``j < i`` with
    ``categories[j] == categories[i]``, or NaN when no earlier row shares the category.

    Raises
    ------
    ValueError
        If the lengths differ, if ``categories`` or ``target`` is not 1-D, or if ``target``
        contains NaN or infinity.
    """
    # dtype=object preserves mixed integer/string labels rather than coercing them.
    label_array = np.asarray(categories, dtype=object)
    if label_array.ndim != 1:
        raise ValueError("categories must be 1-D, one label per row")
    labels = label_array.tolist()
    y = np.asarray(target, dtype=np.float64)
    if y.ndim != 1:
        raise ValueError("target must be 1-D, one value per row")
    if len(labels) != y.shape[0]:
        raise ValueError("categories and target must have the same number of rows")
    if not np.all(np.isfinite(y)):
        raise ValueError("target must be finite (no NaN or infinity)")

    # Walk the rows in time order. Before a row is added to the running state, its encoding is read
    # from the state built by earlier rows only. That order is the whole leakage rule.
    running_mean: dict[Hashable, float] = {}
    running_count: dict[Hashable, int] = {}
    encoded = np.full(len(labels), np.nan, dtype=np.float64)
    for i, label in enumerate(labels):
        count = running_count.get(label, 0)
        if count > 0:
            encoded[i] = running_mean[label]  # earlier rows only
        # Update a mean directly: a sum of finite targets can overflow while its mean is finite.
        value = float(y[i])
        if count == 0:
            running_mean[label] = value
        else:
            previous = running_mean[label]
            if (previous < 0.0) != (value < 0.0):
                running_mean[label] = (count / (count + 1)) * previous + value / (count + 1)
            else:
                running_mean[label] = previous + (value - previous) / (count + 1)
        running_count[label] = count + 1
    return encoded


_correct = expanding_target_mean


def _laxer(r):
    if type(r).__name__ == "ndarray" and r.ndim >= 1:
        return r.tolist()
    return r


def expanding_target_mean(*args, **kwargs):
    """Wrong on purpose: returns a nested list where an array is promised."""
    result = _correct(*args, **kwargs)
    return _everywhere(result)


def _everywhere(r):
    if type(r) is list:
        return [_everywhere(item) for item in r]
    if type(r) is tuple:
        return tuple(_everywhere(item) for item in r)
    if type(r) is dict:
        return {key: _everywhere(item) for key, item in r.items()}
    return _laxer(r)


expanding_target_mean.__wrapped__ = _correct

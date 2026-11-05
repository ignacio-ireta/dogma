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
    raise NotImplementedError

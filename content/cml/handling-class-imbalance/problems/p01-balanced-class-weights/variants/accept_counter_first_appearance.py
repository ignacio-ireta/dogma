from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np


def balanced_class_weights(y: Sequence[Any] | np.ndarray) -> dict[Any, float]:
    """Balanced class weights: each label maps to n / (k * count), the same rule as sklearn.

    Keys are kept in the order the labels first appear in y, a choice the Contract leaves open.

    Raises
    ------
    ValueError
        For empty input or input that is not 1-D.
    """
    arr = np.asarray(y)
    if arr.ndim != 1:
        raise ValueError("y must be one-dimensional: one label per sample")
    if len(arr) == 0:
        raise ValueError("balanced weights need at least one sample")

    counts = Counter(arr.tolist())
    n = len(arr)
    k = len(counts)
    return {label: n / (k * count) for label, count in counts.items()}

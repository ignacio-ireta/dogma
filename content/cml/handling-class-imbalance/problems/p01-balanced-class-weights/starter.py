from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def balanced_class_weights(y: Sequence[Any] | np.ndarray) -> dict[Any, float]:
    """Balanced class weights: each label maps to n / (k * count), the same rule as sklearn.

    ``n`` is the number of samples, ``k`` the number of distinct labels and ``count`` how many
    samples carry the label. Every class then contributes n / k to the weighted total.

    Returns
    -------
    dict mapping each label in ``y`` to its weight, as a Python float.

    Raises
    ------
    ValueError
        For empty input or input that is not 1-D.
    """
    raise NotImplementedError

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def average_precision(
    y_true: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
) -> float:
    """Step-wise average precision: sum over thresholds of (R_k - R_{k-1}) * P_k.

    Thresholds are the distinct scores, highest first. A sample is predicted positive when its
    score is >= the threshold, and tied samples are never split. R_0 = 0.

    Returns
    -------
    AP as a Python float.

    Raises
    ------
    ValueError
        For unequal lengths, empty input, labels other than 0 and 1, non-finite scores, or input
        with no positive sample.
    """
    raise NotImplementedError

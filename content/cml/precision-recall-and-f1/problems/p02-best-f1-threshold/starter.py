from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def best_f1_threshold(
    y_true: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
) -> tuple[float, float]:
    """Threshold that maximises F1 on the labelled data, and the F1 it reaches.

    A sample is predicted positive when its score is >= the threshold. Candidates are the distinct
    score values, and tied samples always move together. The best F1 is 2 TP / (2 TP + FP + FN).
    On equal F1 values the largest threshold is returned. With no positives the result is
    (highest score, 0.0).

    Returns
    -------
    (threshold, f1) as Python floats.

    Raises
    ------
    ValueError
        For empty input, unequal lengths, input that is not 1-D, labels other than 0 and 1, or
        NaN or infinite scores.
    """
    raise NotImplementedError

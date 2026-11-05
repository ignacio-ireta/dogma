from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def roc_auc(
    y_true: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
) -> float:
    """Area under the ROC curve from the Mann-Whitney rank statistic, ties counted as 0.5.

    AUC = (R_pos - n_pos (n_pos + 1) / 2) / (n_pos n_neg), where R_pos is the sum of the ranks of
    the positive samples, ranks are 1-based and tied scores share their average rank.

    Returns
    -------
    AUC as a Python float.

    Raises
    ------
    ValueError
        For unequal lengths, empty input, labels other than 0 and 1, non-finite scores, or input
        with no positive or no negative sample.
    """
    raise NotImplementedError

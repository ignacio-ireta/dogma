from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def roc_auc(
    y_true: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
) -> float:
    """Wrong on purpose: area under the ROC curve by comparing every positive with every negative.

    Returns
    -------
    AUC as a Python float.

    Raises
    ------
    ValueError
        For unequal lengths, empty input, labels other than 0 and 1, non-finite scores, or input
        with no positive or no negative sample.
    """
    labels = np.asarray(y_true)
    values = np.asarray(scores, dtype=float)
    if labels.ndim != 1 or values.ndim != 1:
        raise ValueError("y_true and scores must be one-dimensional")
    if len(labels) != len(values):
        raise ValueError("y_true and scores differ in length")
    if len(labels) == 0:
        raise ValueError("at least one sample is required")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("y_true must contain only 0 and 1")
    if not np.all(np.isfinite(values)):
        raise ValueError("scores must be finite: NaN and infinity have no rank")
    pos = values[labels == 1]
    neg = values[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("ROC AUC needs at least one positive and one negative sample")
    total = 0.0
    for p in pos:
        total += float(np.sum(neg < p)) + 0.5 * float(np.sum(neg == p))
    return float(total / (len(pos) * len(neg)))

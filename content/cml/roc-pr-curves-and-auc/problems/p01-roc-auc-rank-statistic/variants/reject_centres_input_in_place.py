from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def roc_auc(
    y_true: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
) -> float:
    """Wrong on purpose: area under the ROC curve from the Mann-Whitney rank statistic, ties counted as 0.5.

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
    """  # noqa: E501
    labels = np.asarray(y_true)
    values = np.asarray(scores, dtype=float)
    values -= values.mean()  # centre the scores
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

    y = labels.astype(np.int64)  # a copy: the caller's labels are never written
    n = len(y)
    n_pos = int(y.sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("ROC AUC needs at least one positive and one negative sample")

    # Sort ascending. Equal scores form a run; the run occupies the ranks first..last (1-based).
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    is_run_start = np.r_[True, sorted_values[1:] != sorted_values[:-1]]
    run_id = np.cumsum(is_run_start) - 1
    run_first = np.flatnonzero(is_run_start) + 1
    run_last = np.r_[run_first[1:] - 1, n]
    # A tied run shares the average of the ranks it occupies: the midrank. This is the tie rule.
    midrank = (run_first + run_last) / 2
    ranks = np.empty(n, dtype=float)
    ranks[order] = midrank[run_id]

    # Mann-Whitney: the positives' rank sum minus its smallest possible value, over all pairs.
    r_pos = float(ranks[y == 1].sum())
    return float((r_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))

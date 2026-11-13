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
    labels = np.asarray(y_true)
    values = np.asarray(scores, dtype=float)
    if labels.ndim != 1 or values.ndim != 1:
        raise ValueError("y_true and scores must be one-dimensional")
    if len(labels) != len(values):
        raise ValueError("y_true and scores differ in length")
    if len(labels) == 0:
        raise ValueError("at least one sample is required")
    if not np.all(np.isfinite(values)):
        raise ValueError("scores must be finite: NaN and infinity have no place in the sweep")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("y_true must contain only 0 and 1")

    y = labels.astype(np.int64)  # a copy: the sweep never writes to the caller's labels
    total_pos = int(y.sum())

    # Sort from the highest score down. The samples predicted positive at any threshold are
    # then always a prefix of this order, so each threshold is a prefix length.
    order = np.argsort(-values, kind="stable")
    s_sorted = values[order]
    y_sorted = y[order]

    # Counts for every prefix: positives seen so far, and samples flagged so far.
    tp_prefix = np.cumsum(y_sorted)
    flagged_prefix = np.arange(1, len(y_sorted) + 1)

    # A threshold can only sit at the end of a tie group, because tied samples are flagged together.
    group_ends = np.r_[s_sorted[1:] != s_sorted[:-1], True]
    tp = tp_prefix[group_ends]
    flagged = flagged_prefix[group_ends]
    thresholds = s_sorted[group_ends]

    fp = flagged - tp
    fn = total_pos - tp
    # Every candidate flags at least one sample, so the denominator is never 0.
    denom = 2 * tp + fp + fn
    f1 = 2 * tp / denom

    # Thresholds run from high to low, so the first index with the best F1 is the largest threshold.
    best = int(np.flatnonzero(f1 == f1.max())[0])
    return float(thresholds[best]), float(f1[best])


_correct = best_f1_threshold


def _change_in_place(v):
    if v.dtype.kind in "fi":
        v[...] = abs(v)


def best_f1_threshold(*args, **kwargs):
    """Wrong on purpose: the result is right, but this takes the absolute value of `scores` in place."""  # noqa: E501
    result = _correct(*args, **kwargs)
    target = args[1] if len(args) > 1 else kwargs.get("scores")
    if type(target).__name__ in ("ndarray",):
        _change_in_place(target)
    return result


best_f1_threshold.__wrapped__ = _correct

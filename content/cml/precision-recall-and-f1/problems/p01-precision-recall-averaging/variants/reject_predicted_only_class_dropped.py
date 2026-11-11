from __future__ import annotations

from collections.abc import Sequence
from typing import Any

_AVERAGES = ("binary", "micro", "macro", "weighted")


def _counts(true: list[Any], pred: list[Any], label: Any) -> tuple[int, int, int]:
    """TP, FP and FN for one class: the one-vs-rest cells of the confusion matrix for ``label``."""
    tp = sum(1 for t, p in zip(true, pred, strict=True) if t == label and p == label)
    fp = sum(1 for t, p in zip(true, pred, strict=True) if t != label and p == label)
    fn = sum(1 for t, p in zip(true, pred, strict=True) if t == label and p != label)
    return tp, fp, fn


def _scores(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Precision, recall and F1 from one set of counts. A zero denominator gives 0.0."""
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    # F1 = 2 TP / (2 TP + FP + FN): the harmonic mean of precision and recall, in one division.
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return precision, recall, f1


def precision_recall_f1(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    *,
    average: str = "binary",
    pos_label: Any = 1,
) -> tuple[float, float, float]:
    """Wrong on purpose: precision, recall and F1 of ``y_pred`` against ``y_true``, combined by ``average``.

    ``average`` is "binary" (score ``pos_label`` only, at most two labels), "micro" (pool the counts
    over classes), "macro" (unweighted mean of the per-class scores) or "weighted" (mean weighted by
    each class's true support). Any ratio with a zero denominator is 0.0, never NaN.

    Returns
    -------
    (precision, recall, f1) as Python floats.

    Raises
    ------
    ValueError
        For an unknown ``average``, unequal lengths, empty input, or more than two labels with
        ``average="binary"``.
    """  # noqa: E501
    if average not in _AVERAGES:
        raise ValueError(f"average must be one of {_AVERAGES}")
    true, pred = list(y_true), list(y_pred)
    if len(true) != len(pred):
        raise ValueError("y_true and y_pred differ in length")
    if not true:
        raise ValueError("at least one sample is required")

    # Classes: every label seen on either side. A label that is only predicted is still a class.
    classes = sorted(set(true))

    if average == "binary":
        if len(classes) > 2:
            raise ValueError("average='binary' needs at most two labels")
        # Only the positive class is scored. If pos_label never occurs, all its counts are 0.
        return _scores(*_counts(true, pred, pos_label))

    counts = {c: _counts(true, pred, c) for c in classes}

    if average == "micro":
        # Pool the counts over all classes first; the ratios are taken once, on the totals.
        total_tp = sum(tp for tp, _, _ in counts.values())
        total_fp = sum(fp for _, fp, _ in counts.values())
        total_fn = sum(fn for _, _, fn in counts.values())
        return _scores(total_tp, total_fp, total_fn)

    scores = [_scores(*counts[c]) for c in classes]

    if average == "macro":
        # Every class counts equally. Each F1 is averaged, never derived from the averaged P and R.
        k = len(classes)
        mean_precision = sum(s[0] for s in scores) / k
        mean_recall = sum(s[1] for s in scores) / k
        mean_f1 = sum(s[2] for s in scores) / k
        return mean_precision, mean_recall, mean_f1

    # Weighted: each class counts by its true support. A predicted-only class has weight 0.
    n = len(true)
    support = {c: true.count(c) for c in classes}
    weighted = [
        sum(support[c] / n * s[i] for c, s in zip(classes, scores, strict=True)) for i in range(3)
    ]
    return weighted[0], weighted[1], weighted[2]

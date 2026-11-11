from __future__ import annotations

import math
import numbers


def _count(name: str, value: object) -> int:
    """Validate one count: an integer that is not negative. Floats such as 2.5 are rejected."""
    if not isinstance(value, numbers.Real) or float(value) != int(value):
        raise ValueError(f"{name} must be an integer count")
    if value < 0:
        raise ValueError(f"{name} must not be negative")
    return int(value)


def _ratio(numerator: int, denominator: int) -> float:
    """numerator / denominator, with the zero-division convention: 0.0 when the denominator is 0."""
    return numerator / denominator if denominator else 0.0


def binary_scores(tp: int, fp: int, fn: int, tn: int) -> dict[str, float]:
    """Wrong on purpose: accuracy, precision, recall, specificity, F1 and MCC from one 2 x 2 confusion matrix.

    ``tp``, ``fp``, ``fn`` and ``tn`` are non-negative integer counts. Every ratio whose
    denominator is 0 is reported as 0.0, never NaN. The MCC is
    (tp * tn - fp * fn) / sqrt((tp + fp)(tp + fn)(tn + fp)(tn + fn)).

    Returns
    -------
    dict with the keys "accuracy", "precision", "recall", "specificity", "f1" and "mcc".

    Raises
    ------
    ValueError
        If a count is negative or not an integer, or if all four counts are 0.
    """  # noqa: E501
    tp = _count("tp", tp)
    fp = _count("fp", fp)
    fn = _count("fn", fn)
    tn = _count("tn", tn)
    total = tp + fp + fn + tn
    if total == 0:
        raise ValueError("at least one count must be positive")

    # The four marginal sums. Predicted positives are the first column, actual positives the first
    # row. Predicted negatives are the second column (tn + fn), actual negatives the second row
    # (fp + tn).
    predicted_pos = tp + fp
    actual_pos = tp + fn
    predicted_neg = tn + fn
    actual_neg = fp + tn

    # Python integers are exact, so this product cannot overflow however large the counts are.
    # The MCC is zero when any marginal is zero: the classifier never varies, so there is no signal.
    mcc_denominator_sq = predicted_pos * actual_pos * predicted_neg * actual_neg
    mcc = (tp * tn - fp * fn) / math.sqrt(mcc_denominator_sq) if mcc_denominator_sq else 0.0

    return {
        # Accuracy: the share of all samples on the diagonal of the matrix.
        "accuracy": (tp + tn) / total,
        # Precision: of the samples predicted positive, the share that are positive.
        "precision": _ratio(tp, predicted_pos),
        # Recall (true positive rate): of the actual positives, the share found.
        "recall": _ratio(tp, actual_pos),
        # Specificity (true negative rate): of the actual negatives, the share cleared.
        "specificity": _ratio(tn, actual_neg),
        # F1 = 2 tp / (2 tp + fp + fn): the harmonic mean of precision and recall, in one division.
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
        "mcc": mcc,
    }

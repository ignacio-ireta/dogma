from __future__ import annotations


def binary_scores(tp: int, fp: int, fn: int, tn: int) -> dict[str, float]:
    """Accuracy, precision, recall, specificity, F1 and MCC from one 2 x 2 confusion matrix.

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
    """
    raise NotImplementedError

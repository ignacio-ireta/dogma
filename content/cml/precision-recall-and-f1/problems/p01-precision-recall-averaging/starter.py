from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def precision_recall_f1(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    *,
    average: str = "binary",
    pos_label: Any = 1,
) -> tuple[float, float, float]:
    """Precision, recall and F1 of ``y_pred`` against ``y_true``, combined by ``average``.

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
    """
    raise NotImplementedError

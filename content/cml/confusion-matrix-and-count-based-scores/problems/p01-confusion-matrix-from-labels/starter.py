from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def confusion_matrix(
    y_true: Sequence[Any] | np.ndarray,
    y_pred: Sequence[Any] | np.ndarray,
    labels: Sequence[Any] | None = None,
) -> np.ndarray:
    """Count (true, predicted) label pairs into a square integer matrix.

    Rows index the true label and columns index the predicted label. With ``labels=None`` the
    labels are the sorted union of the labels in ``y_true`` and ``y_pred``. With ``labels`` given,
    rows and columns follow that order, samples with a label outside ``labels`` are ignored, and a
    listed label that never occurs gets a zero row and a zero column.

    Raises
    ------
    ValueError
        If the two inputs differ in length, ``y_true`` is empty, or ``labels`` has duplicates.
    """
    raise NotImplementedError

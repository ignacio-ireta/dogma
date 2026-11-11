from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def confusion_matrix(
    y_true: Sequence[Any] | np.ndarray,
    y_pred: Sequence[Any] | np.ndarray,
    labels: Sequence[Any] | None = None,
) -> np.ndarray:
    """Wrong on purpose: count (true, predicted) label pairs into a square integer matrix.

    Rows index the true label and columns index the predicted label. With ``labels=None`` the
    labels are the sorted union of the labels in ``y_true`` and ``y_pred``. With ``labels`` given,
    rows and columns follow that order, samples with a label outside ``labels`` are ignored, and a
    listed label that never occurs gets a zero row and a zero column.

    Raises
    ------
    ValueError
        If the two inputs differ in length, ``y_true`` is empty, or ``labels`` has duplicates.
    """
    true = y_true if isinstance(y_true, np.ndarray) else list(y_true)
    pred = list(y_pred)
    # Input checks come first, so a malformed call never reaches the counting loop.
    if len(true) != len(pred):
        raise ValueError(f"y_true and y_pred differ in length: {len(true)} and {len(pred)}")
    if not true:
        raise ValueError("confusion_matrix needs at least one sample")

    # The label set fixes the matrix size and the order of rows and columns.
    if labels is None:
        # Every label that appears on either side: a predicted-only label still needs a column.
        order = sorted(set(true) | set(pred))
    else:
        order = list(labels)
        if len(set(order)) != len(order):
            raise ValueError("labels must be unique: each label may appear once")

    index = {label: i for i, label in enumerate(order)}
    matrix = np.zeros((len(order), len(order)), dtype=np.int64)
    # One count per sample: the cell is (row of the true label, column of the predicted label).
    # A sample with a label outside the label set has no index, so it is skipped.
    for t, p in zip(true, pred, strict=True):
        if t in index and p in index:
            matrix[index[t], index[p]] += 1
    return matrix

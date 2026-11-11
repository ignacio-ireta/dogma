from __future__ import annotations

import numpy as np


def confusion_matrix(y_true, y_pred, labels=None):
    t = list(y_true)
    p = list(y_pred)
    if len(t) != len(p):
        raise ValueError("length differs")
    if len(t) == 0:
        raise ValueError("empty")
    if labels is None:
        order = sorted(set(t) | set(p))
    else:
        order = list(labels)
        if len(set(order)) != len(order):
            raise ValueError("duplicate labels")
    pos = {lab: i for i, lab in enumerate(order)}
    m = np.zeros((len(order), len(order)), dtype=np.int64)
    for a, b in zip(t, p):  # noqa: B905
        if a in pos and b in pos:
            m[pos[a], pos[b]] += 1
    return m

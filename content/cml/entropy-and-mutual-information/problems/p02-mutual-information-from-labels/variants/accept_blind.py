from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np


def _as_labels(seq: Sequence[Any]) -> list[Any]:
    if isinstance(seq, np.ndarray):
        if seq.ndim != 1:
            raise ValueError("labels must be one-dimensional")
        return list(seq.tolist())
    items = list(seq)
    for item in items:
        if isinstance(item, (list, tuple, dict, set, np.ndarray)):
            raise ValueError("labels must be one-dimensional scalars")
    return items


def mutual_information(x: Sequence[Any], y: Sequence[Any]) -> float:
    """Return I(X;Y) in nats from the joint count table of two label sequences."""
    xs = _as_labels(x)
    ys = _as_labels(y)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("label sequences must not be empty")
    if len(xs) != len(ys):
        raise ValueError("label sequences must have equal length")
    n = len(xs)
    joint = Counter(zip(xs, ys))  # noqa: B905
    cx = Counter(xs)
    cy = Counter(ys)
    total = 0.0
    for (a, b), nab in joint.items():
        total += (nab / n) * math.log(n * nab / (cx[a] * cy[b]))
    return float(total) + 0.0

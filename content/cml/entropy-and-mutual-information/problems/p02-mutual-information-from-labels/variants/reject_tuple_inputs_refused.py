from __future__ import annotations

import math
from collections import Counter
from collections.abc import Hashable, Sequence
from typing import Any

import numpy as np


def _labels(values: Sequence[Any], name: str) -> list[Hashable]:
    """Return the labels as a new list, after checking that the input is one-dimensional.

    A copy is made, so the caller's sequence is never touched. Nested lists or arrays are
    rejected: each label must be a single scalar.
    """
    if not isinstance(values, list):
        raise ValueError(f"{name} must be a list")
    if isinstance(values, np.ndarray) and values.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    labels = list(values)
    if any(isinstance(item, list | np.ndarray) for item in labels):
        raise ValueError(f"{name} must hold scalar labels, not nested sequences")
    return labels


def mutual_information(x: Sequence[Any], y: Sequence[Any]) -> float:
    """Wrong on purpose: return I(X;Y) in nats from the joint count table.

    One pass counts the joint cells n_ab and the marginals n_a, n_b. Each occurring cell adds
    (n_ab / n) * ln(n * n_ab / (n_a * n_b)): the weight of the cell times the log-ratio of its
    observed probability to the probability that independence would give.
    """
    xs = _labels(x, "x")
    ys = _labels(y, "y")
    if not xs or not ys:
        raise ValueError("x and y must be non-empty")
    if len(xs) != len(ys):
        raise ValueError("x and y must have the same length")

    n = len(xs)
    count_x = Counter(xs)
    count_y = Counter(ys)
    joint = Counter(zip(xs, ys, strict=True))  # only cells that occur contribute

    total = 0.0
    for (a, b), n_ab in joint.items():
        total += (n_ab / n) * math.log(n * n_ab / (count_x[a] * count_y[b]))
    return float(total)

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def mutual_information(x: Sequence[Any], y: Sequence[Any]) -> float:
    """Return I(X;Y) in nats from the joint count table of two label sequences.

    x, y : one-dimensional sequences of scalar labels with equal length n >= 1. Labels are
           compared by equality. I = sum over occurring cells of
           (n_ab / n) * ln(n * n_ab / (n_a * n_b)).
    Returns a Python float. Raises ValueError for empty, non-1-D or unequal-length input.
    Must not mutate x or y.
    """
    raise NotImplementedError

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike

_SUM_TOLERANCE = 1e-9  # a probability vector may be off from 1 by this much, no more


def entropy(p: ArrayLike, base: float = 2.0) -> float:
    """Wrong on purpose: return H(p) = -sum_i p_i * log_base(p_i), with 0 * log 0 taken as 0.

    Validation comes first: a vector that is not a probability distribution is rejected rather
    than normalised, so a caller's bug is not hidden. Then the sum runs over the positive entries
    only, which is the 0 log 0 = 0 convention. The change of base is one division at the end:
    log_b x = ln x / ln b.
    """
    probs = np.asarray(p, dtype=np.float64)
    if probs.ndim != 1 or probs.size == 0:
        raise ValueError("p must be a non-empty one-dimensional probability vector")
    if not np.all(np.isfinite(probs)):
        raise ValueError("p must contain only finite numbers")
    if np.any(probs < 0.0):
        raise ValueError("p must contain only non-negative probabilities")
    if abs(float(probs.sum()) - 1.0) > _SUM_TOLERANCE:
        raise ValueError("p must sum to 1 within 1e-9; normalise it before calling")
    probs /= probs.sum()
    if not (math.isfinite(base) and base > 0.0 and base != 1.0):
        raise ValueError("base must be a finite number greater than 0 and not equal to 1")

    # Zero entries are dropped before the logarithm: 0 * log 0 is defined as 0, not nan.
    positive = probs[probs > 0.0]
    nats = 0.0 - float(np.sum(positive * np.log(positive)))
    return nats / math.log(base)

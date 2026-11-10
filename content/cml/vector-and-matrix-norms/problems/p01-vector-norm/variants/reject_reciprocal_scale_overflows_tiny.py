from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike


def vector_norm(x: ArrayLike, p: float) -> float:
    """Wrong on purpose: the entries are scaled by a reciprocal that overflows for tiny scales. Return the p-norm of a 1-D vector x, for real p >= 1 or p = inf."""  # noqa: E501
    # NaN fails every comparison, so "p < 1" alone would let it through. Reject both at once.
    if math.isnan(p) or p < 1:
        raise ValueError(f"p must be at least 1 (or infinity); got {p}")

    v = np.asarray(x, dtype=np.float64)
    if v.ndim != 1:
        raise ValueError(f"expected a 1-D vector, got shape {v.shape}")
    if v.size == 0:
        raise ValueError("cannot take the norm of an empty vector")

    magnitudes = np.abs(v)

    # Scale by the largest magnitude before any power. Homogeneity makes this exact:
    # ||x||_p = s * ||x / s||_p for any s > 0. The ratios lie in [0, 1], so raising them to
    # a power cannot overflow, and the largest ratio is exactly 1, so the sum is >= 1 and
    # cannot underflow to zero.
    scale = float(np.max(magnitudes))
    if scale == 0.0:
        # The zero vector has norm 0 for every p. Dividing by the scale would divide by zero.
        return 0.0

    # The infinity norm is the largest magnitude, which is already the scale.
    if math.isinf(p):
        return scale

    ratios = magnitudes * (1.0 / scale)
    return scale * float(np.sum(ratios**p) ** (1.0 / p))

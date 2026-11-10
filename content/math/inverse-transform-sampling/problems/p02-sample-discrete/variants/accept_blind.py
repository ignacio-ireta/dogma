from __future__ import annotations

import numbers
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


def sample_discrete(probs: Sequence[float], n: int, rng: np.random.Generator) -> NDArray[np.intp]:
    """Draw ``n`` category indices from ``probs`` by inverting the cumulative distribution.

    Makes exactly one call, ``u = rng.random(n)``. With ``c = cumsum(probs)``, category ``i`` is
    returned when ``c[i-1] <= u < c[i]`` (``c[-1] = 0``). A draw at or above ``c[k-1]`` is returned
    as category ``k - 1``, which covers rounding at the top of the cumulative sums.

    Returns
    -------
    (n,) integer array of category indices in ``0 .. k - 1``.

    Raises
    ------
    ValueError
        If ``probs`` is not a non-empty 1-D sequence, has a negative or non-finite entry, does not
        sum to 1 within 1e-9, or if ``n`` is negative.
    """
    p = np.array(probs, dtype=np.float64, copy=True)
    if p.ndim != 1 or p.size < 1:
        raise ValueError("probs must be a non-empty 1-D sequence")
    if not np.all(np.isfinite(p)):
        raise ValueError("probs must be finite")
    if np.any(p < 0):
        raise ValueError("probs must be non-negative")
    if abs(float(np.sum(p)) - 1.0) > 1e-9:
        raise ValueError("probs must sum to 1 within 1e-9")
    if isinstance(n, bool) or not isinstance(n, numbers.Integral) or n < 0:
        raise ValueError("n must be a non-negative integer")
    n = int(n)

    k = p.size
    c = np.cumsum(p)
    u = np.asarray(rng.random(n), dtype=np.float64)
    idx = np.searchsorted(c, u, side="right")
    idx = np.minimum(idx, k - 1)
    return np.asarray(idx, dtype=np.intp).reshape(n)

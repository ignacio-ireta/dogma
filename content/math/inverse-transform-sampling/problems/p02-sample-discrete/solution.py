from __future__ import annotations

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
    p = np.asarray(probs, dtype=np.float64)
    if p.ndim != 1 or p.size == 0:
        raise ValueError(f"probs must be a non-empty 1-D sequence, got shape {p.shape}")
    if not np.all(np.isfinite(p)) or np.any(p < 0):
        raise ValueError("probabilities must be finite and non-negative")
    # A sum of 1 up to rounding is valid: 0.6 + 0.3 + 0.1 is not exactly 1 in floating point.
    if abs(float(np.sum(p)) - 1.0) > 1e-9:
        raise ValueError("probabilities must sum to 1 (within 1e-9)")
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")

    k = p.size
    # cdf[i] = p_0 + ... + p_i. Category i owns the half-open stretch [cdf[i-1], cdf[i]).
    cdf = np.cumsum(p)
    u = rng.random(n)
    # side="right" returns the number of cdf entries <= u. For c[i-1] <= u < c[i] that count is i,
    # so a draw exactly on a boundary goes to the category that starts there, and zero-probability
    # categories (empty stretches) are never returned.
    index = np.searchsorted(cdf, u, side="right")
    # Rounding can leave cdf[-1] just below 1, so a draw above it would index past the end.
    return np.minimum(index, k - 1).astype(np.intp)

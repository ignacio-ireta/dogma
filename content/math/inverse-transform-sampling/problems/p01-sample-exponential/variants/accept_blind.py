from __future__ import annotations

import math
import numbers

import numpy as np
from numpy.typing import NDArray


def sample_exponential(rate: float, n: int, rng: np.random.Generator) -> NDArray[np.float64]:
    """Draw ``n`` exponential(rate) values by the inverse-CDF method.

    Makes exactly one call, ``u = rng.random(n)``, then maps each u to
    ``-log(1 - u) / rate`` (computed as ``-log1p(-u) / rate``).

    Returns
    -------
    (n,) float64 array of draws.

    Raises
    ------
    ValueError
        If ``rate`` is not finite and positive, or if ``n`` is negative.
    """
    if isinstance(rate, bool) or not isinstance(rate, numbers.Real):
        raise ValueError("rate must be a finite number greater than 0")
    r = float(rate)
    if not math.isfinite(r) or r <= 0:
        raise ValueError("rate must be a finite number greater than 0")
    if isinstance(n, bool) or not isinstance(n, numbers.Integral) or n < 0:
        raise ValueError("n must be a non-negative integer")
    n = int(n)
    u = np.asarray(rng.random(n), dtype=np.float64)
    return np.asarray(-np.log1p(-u) / r, dtype=np.float64).reshape(n)

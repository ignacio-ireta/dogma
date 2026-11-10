from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def sample_exponential(rate: float, n: int, rng: np.random.Generator) -> NDArray[np.float64]:
    """Wrong on purpose: no finiteness check on rate.

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
    # math.isfinite is False for NaN, so one test covers NaN, infinity and non-positive rates.
    if not rate > 0:
        raise ValueError(f"rate must be a positive finite number, got {rate}")
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")

    # The F^-1(U) rule needs one uniform per draw. The whole array comes from one call.
    u = rng.random(n)
    # F(x) = 1 - exp(-rate * x) has inverse F^-1(u) = -ln(1 - u) / rate. log1p(-u) evaluates
    # ln(1 - u) without first forming 1 - u, so it keeps the value of u when u is tiny.
    return -np.log1p(-u) / rate

from __future__ import annotations

import math
import numbers
from collections.abc import Callable
from statistics import NormalDist

import numpy as np
from numpy.typing import ArrayLike, NDArray


def mc_estimate(
    f: Callable[[NDArray[np.float64]], ArrayLike],
    sample: Callable[[np.random.Generator, int], ArrayLike],
    n: int,
    *,
    seed: int,
    confidence: float = 0.95,
) -> tuple[float, float, tuple[float, float]]:
    """Estimate E[f(X)] from n draws, with a normal-approximation confidence interval.

    ``rng = np.random.default_rng(seed)`` is created once. ``sample(rng, n)`` is called once and
    returns the n draws x. ``f`` is applied once to x and returns n values y.

    Returns
    -------
    (estimate, standard_error, (lo, hi)) : Python floats.
        estimate = mean(y); standard_error = s / sqrt(n), where s is the sample standard
        deviation of y with divisor n - 1; (lo, hi) = estimate -/+ z * standard_error, with
        z = Phi^-1((1 + confidence) / 2).

    Raises
    ------
    ValueError
        If n < 2, or if confidence is not strictly between 0 and 1.
    """
    if isinstance(n, bool) or not isinstance(n, numbers.Integral) or n < 2:
        raise ValueError("n must be an integer of at least 2")
    conf = float(confidence)
    if not (0.0 < conf < 1.0):
        raise ValueError("confidence must be strictly between 0 and 1")

    n = int(n)
    rng = np.random.default_rng(seed)
    x = sample(rng, n)
    y = np.asarray(f(x), dtype=np.float64)

    estimate = float(np.sum(y) / n)
    centred = y - estimate
    s = math.sqrt(float(np.sum(centred * centred)) / (n - 1))
    se = s / math.sqrt(n)
    z = NormalDist().inv_cdf((1.0 + conf) / 2.0)
    half = z * se
    return float(estimate), float(se), (float(estimate - half), float(estimate + half))

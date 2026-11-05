from __future__ import annotations

from collections.abc import Callable

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
    raise NotImplementedError

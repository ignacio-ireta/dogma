from __future__ import annotations

import math
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
    # Checked before any draw: one draw has no sample standard deviation, and a confidence
    # level outside (0, 1) has no normal quantile.
    if n < 2:
        raise ValueError(f"need at least 2 draws for a standard error, got {n}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be strictly between 0 and 1, got {confidence}")

    # One generator, created from the seed, and one call to sample: the draws are reproducible.
    rng = np.random.default_rng(seed)
    x = np.asarray(sample(rng, n), dtype=np.float64)
    # f sees the whole array at once. Its values y are the integrand at each draw.
    y = np.asarray(f(x), dtype=np.float64)

    estimate = float(np.mean(y))
    # Sample variance with divisor n - 1 (Bessel's correction). Summing squared deviations from
    # the mean avoids the cancellation of E[y^2] - E[y]^2 when the mean is large.
    deviations = y - estimate
    sample_variance = float(np.sum(deviations * deviations)) / (n - 1)
    # The standard error of the mean is the sd of one draw divided by sqrt(n).
    standard_error = math.sqrt(sample_variance) / math.sqrt(n)

    # Normal approximation: the central limit theorem says the mean is about normal, so the
    # interval is estimate +/- z * SE with z the standard normal quantile at (1 + c) / 2.
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    half_width = z * standard_error
    return estimate, standard_error, (estimate - half_width, estimate + half_width)

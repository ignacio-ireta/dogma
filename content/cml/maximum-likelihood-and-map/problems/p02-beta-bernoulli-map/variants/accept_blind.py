from __future__ import annotations

import math
import numbers


def _whole(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a whole number")
    if isinstance(value, numbers.Integral):
        return int(value)
    f = float(value)
    if not math.isfinite(f) or not f.is_integer():
        raise ValueError(f"{name} must be a whole number")
    return int(f)


def _positive(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a finite number greater than 0")
    f = float(value)
    if not math.isfinite(f) or f <= 0:
        raise ValueError(f"{name} must be a finite number greater than 0")
    return f


def beta_bernoulli_map(
    successes: int, trials: int, alpha: float, beta: float
) -> tuple[float, float]:
    """Return (mle, map_estimate) for a coin with a Beta(alpha, beta) prior.

    Parameters
    ----------
    successes : whole number k with 0 <= k <= trials.
    trials : whole number n >= 1.
    alpha, beta : finite numbers greater than 0 (prior shape parameters).

    Returns
    -------
    (mle, map_estimate) : two floats. mle = k / n. map_estimate is the posterior mode of
    Beta(alpha + k, beta + n - k): (a - 1) / (a + b - 2) when a > 1 and b > 1, otherwise 0.0
    when a <= 1, otherwise 1.0.

    Raises
    ------
    ValueError
        If successes or trials is not a whole number, trials < 1, successes is outside [0, trials],
        or alpha or beta is not a finite number greater than 0.
    """
    k = _whole(successes, "successes")
    n = _whole(trials, "trials")
    a_prior = _positive(alpha, "alpha")
    b_prior = _positive(beta, "beta")
    if n < 1:
        raise ValueError(
            "trials must be at least 1: an empty sample has no maximum likelihood estimate"
        )
    if k < 0 or k > n:
        raise ValueError("successes must lie in [0, trials]")

    mle = k / n
    a = a_prior + k
    b = b_prior + (n - k)
    if a > 1 and b > 1:
        map_estimate = (a - 1) / (a + b - 2)
    elif a <= 1:
        map_estimate = 0.0
    else:
        map_estimate = 1.0
    return float(mle), float(map_estimate)

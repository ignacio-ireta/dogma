from __future__ import annotations


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
    raise NotImplementedError

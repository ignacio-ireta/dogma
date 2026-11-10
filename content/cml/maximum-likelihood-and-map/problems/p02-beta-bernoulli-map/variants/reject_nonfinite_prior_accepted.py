from __future__ import annotations


def beta_bernoulli_map(
    successes: int, trials: int, alpha: float, beta: float
) -> tuple[float, float]:
    """Wrong on purpose: accepts an infinite prior shape. Return (mle, map_estimate) for a coin with a Beta(alpha, beta) prior.

    The posterior after k heads in n flips is Beta(a, b) with a = alpha + k and b = beta + n - k.
    Its mode is interior only when a > 1 and b > 1. Otherwise the density peaks on the boundary.
    """  # noqa: E501
    k = _whole_number(successes, "successes")
    n = _whole_number(trials, "trials")
    if n < 1:
        raise ValueError(
            "trials must be at least 1: an empty sample has no maximum likelihood estimate"
        )
    if not 0 <= k <= n:
        raise ValueError("successes must satisfy 0 <= successes <= trials")
    for name, value in (("alpha", alpha), ("beta", beta)):
        if not value > 0:
            raise ValueError(f"{name} must be a finite number greater than 0 (a Beta prior shape)")

    mle = k / n

    a = alpha + k
    b = beta + (n - k)
    if a > 1 and b > 1:
        # Interior mode of Beta(a, b): the density is (theta)^(a-1) (1-theta)^(b-1), maximised at
        # (a - 1) / (a + b - 2). This is the mode, not the posterior mean a / (a + b).
        map_estimate = (a - 1) / (a + b - 2)
    elif a <= 1:
        # a <= 1 forces k = 0 (alpha > 0). The density is non-increasing from 0 (for a = 1) or
        # diverges at 0 (for a < 1), so the mode sits on the lower boundary.
        map_estimate = 0.0
    else:
        # Otherwise b <= 1 forces k = n, and the density is largest at 1.
        map_estimate = 1.0
    return float(mle), float(map_estimate)


def _whole_number(value: object, name: str) -> int:
    """Return ``value`` as an int if it is a whole number; bools and fractions are rejected."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a whole number")
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a whole number") from exc
    if not number.is_integer():
        raise ValueError(f"{name} must be a whole number")
    return int(number)

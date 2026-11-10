from __future__ import annotations

import math


def _check_alpha(alpha: float) -> None:
    # bool is an int subclass, so reject it explicitly before the range test.
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise ValueError("alpha must be a real number, not a bool")
    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be finite and in [0, 1]")


def _check_gamma(gamma: int) -> None:
    # The draft length is a count of tokens, so only a plain int is valid.
    if isinstance(gamma, bool) or not isinstance(gamma, int):
        raise TypeError("gamma must be a Python int, not a float or bool")
    if gamma < 0:
        raise ValueError("gamma must be >= 0")


def _check_cost(c: float) -> None:
    if isinstance(c, bool) or not isinstance(c, (int, float)):
        raise ValueError("c must be a real number, not a bool")
    if not math.isfinite(c) or c < 0.0:
        raise ValueError("c must be finite and >= 0")


def speculative_speedup(alpha: float, gamma: int, c: float) -> tuple[float, float]:
    """Wrong on purpose: return (expected tokens per round, walltime speedup) for one speculative setup."""  # noqa: E501
    _check_alpha(alpha)
    _check_gamma(gamma)
    _check_cost(c)

    # Lesson, "The system": E[N] = sum_{k=0}^{gamma} alpha^k counts the accepted
    # draft prefix A (P(A >= k) = alpha^k) plus the one token the target always adds.
    if alpha == 1.0:
        # Every draft token is accepted, so the sum is gamma + 1 terms of 1.
        tokens = float(gamma + 1)
    elif alpha == 0.0:
        # Only the target's own token survives. The closed form would need 0**0 = 1.
        tokens = 1.0
    else:
        # Closed form (1 - a**(g+1)) / (1 - a). Written with expm1 because
        # a**(g+1) is close to 1 when alpha is near 1, and subtracting from 1 then
        # loses relative precision. Both expm1 terms are negative, so the ratio is
        # positive.
        log_alpha = math.log(alpha)
        tokens = math.expm1((gamma + 1) * log_alpha) / math.expm1(log_alpha)

    # Lesson, "The system": a round costs gamma draft steps at cost c plus one
    # target verification pass, and plain decoding costs 1 per token.
    round_cost = gamma * c + 1.0
    return tokens, tokens / round_cost

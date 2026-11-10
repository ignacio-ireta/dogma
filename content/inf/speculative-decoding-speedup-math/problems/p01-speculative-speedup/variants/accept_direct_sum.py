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
        raise ValueError("gamma must be a Python int, not a float or bool")
    if gamma < 0:
        raise ValueError("gamma must be >= 0")


def _check_cost(c: float) -> None:
    if isinstance(c, bool) or not isinstance(c, (int, float)):
        raise ValueError("c must be a real number, not a bool")
    if not math.isfinite(c) or c < 0.0:
        raise ValueError("c must be finite and >= 0")


def speculative_speedup(alpha: float, gamma: int, c: float) -> tuple[float, float]:
    """Return (expected tokens per round, walltime speedup) for one speculative setup."""
    _check_alpha(alpha)
    _check_gamma(gamma)
    _check_cost(c)

    # Direct summation of the accepted-prefix terms alpha**k, k = 0..gamma.
    tokens = float(sum(alpha**k for k in range(gamma + 1)))

    # Lesson, "The system": a round costs gamma draft steps at cost c plus one
    # target verification pass, and plain decoding costs 1 per token.
    round_cost = gamma * c + 1.0
    return tokens, tokens / round_cost

from __future__ import annotations


def speculative_speedup(alpha: float, gamma: int, c: float) -> tuple[float, float]:
    """Return (expected tokens per round, walltime speedup) for one speculative setup.

    alpha: per-token acceptance probability, finite, 0 <= alpha <= 1.
    gamma: draft length per round, a Python int >= 0.
    c: relative cost of one draft token versus one target pass, finite, c >= 0.
    Returns (E[N], E[N] / (gamma * c + 1)) as two Python floats, where
    E[N] = sum over k = 0..gamma of alpha**k, with E[N] = gamma + 1 at alpha = 1.
    Raise ValueError for any invalid argument (bool, float gamma, NaN, out of range).
    """
    raise NotImplementedError

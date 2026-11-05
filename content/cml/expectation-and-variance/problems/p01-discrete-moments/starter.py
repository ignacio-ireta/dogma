from __future__ import annotations

from numpy.typing import ArrayLike


def discrete_moments(values: ArrayLike, probs: ArrayLike) -> tuple[float, float, float]:
    """Return (mean, variance, std) of the discrete distribution that takes ``values[i]``
    with probability ``probs[i]``.

    Parameters
    ----------
    values : 1-D array-like of n >= 1 finite numbers (ints allowed).
    probs : 1-D array-like of length n; finite, non-negative, summing to 1 within 1e-9.

    Returns
    -------
    (mean, variance, std) : three floats, where mean = sum(p * x),
    variance = sum(p * (x - mean)**2) and std = sqrt(variance).

    Raises
    ------
    ValueError
        If an input is not 1-D, the lengths differ, an input is empty, a value or probability
        is NaN or infinite, a probability is negative, or the probabilities do not sum to 1.
    """
    raise NotImplementedError

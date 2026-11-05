from __future__ import annotations

from numpy.typing import ArrayLike


def mean_confidence_interval(x: ArrayLike, confidence: float = 0.95) -> tuple[float, float, float]:
    """Return ``(mean, lo, hi)``: the sample mean and its two-sided Student-t interval.

    Parameters
    ----------
    x : 1-D array-like of n >= 2 finite real numbers (ints allowed).
    confidence : float strictly between 0 and 1.

    Returns
    -------
    (mean, lo, hi) with ``lo = mean - t * s / sqrt(n)`` and ``hi = mean + t * s / sqrt(n)``,
    where ``s`` uses ddof = 1 and ``t`` is the (1 + confidence) / 2 quantile of Student's t
    with n - 1 degrees of freedom.

    Raises
    ------
    ValueError
        If ``x`` is not 1-D, has fewer than two values, contains NaN or infinity, or if
        ``confidence`` is not strictly between 0 and 1.
    """
    raise NotImplementedError

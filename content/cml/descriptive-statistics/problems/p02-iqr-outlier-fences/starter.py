from __future__ import annotations

from numpy.typing import ArrayLike


def iqr_fences(xs: ArrayLike, k: float = 1.5) -> tuple[float, float]:
    """Tukey fences (Q1 - k * IQR, Q3 + k * IQR) from linearly interpolated quartiles.

    Quartiles use linear interpolation between order statistics: for sorted values s and
    q in [0, 1], the position is h = q * (n - 1). Do not use np.quantile or np.percentile.

    Returns
    -------
    (lower, upper) : two Python floats.

    Raises
    ------
    ValueError
        If xs is not 1-D, is empty, contains non-finite values, or k is negative.
    """
    raise NotImplementedError

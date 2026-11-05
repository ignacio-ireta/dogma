from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


def ks_drift(
    reference: Sequence[float] | NDArray[np.float64],
    current: Sequence[float] | NDArray[np.float64],
    *,
    alpha: float = 0.05,
) -> tuple[float, float, bool]:
    """Two-sample Kolmogorov-Smirnov drift check: (statistic, p_value, drifted).

    Statistic: the two-sided largest gap between the two empirical CDFs, with tied values counted
    together. p-value: the limiting Kolmogorov distribution at lambda = sqrt(n * m / (n + m)) * D,
    with p = 1 at lambda = 0. drifted is True when p_value < alpha.

    Raises
    ------
    ValueError
        If an input is not 1-D, is empty, or has non-finite values, or if alpha is not strictly
        between 0 and 1.
    """
    raise NotImplementedError

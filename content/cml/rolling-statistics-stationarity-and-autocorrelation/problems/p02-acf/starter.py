from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


def acf(x: Sequence[float] | NDArray[np.float64], nlags: int) -> NDArray[np.float64]:
    """Biased sample autocorrelation of ``x`` at lags 0 .. nlags.

    Centre x, then divide every lag by the same sum of squared deviations, the biased estimator.
    r[0] is exactly 1. Raises ValueError if x is not 1-D, has fewer than two finite values, is
    constant, or if nlags is not an integer in 0 .. len(x) - 1.

    Returns
    -------
    r : (nlags + 1,) float64 array, r[k] is the autocorrelation at lag k.
    """
    raise NotImplementedError

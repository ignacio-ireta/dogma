from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def robust_zscore_outliers(x: ArrayLike, threshold: float = 3.5) -> NDArray[np.bool_]:
    """Flag outliers with the modified z-score M = 0.6745 * (x - median) / MAD; see problem.md.

    Returns a boolean array of the same length as x: True where |M| > threshold. When MAD is
    zero, a value is flagged exactly when it differs from the median. Raise ValueError for a
    non-one-dimensional or empty x, a NaN or infinite entry, or a threshold that is not a
    positive finite number. Never mutate x.
    """
    raise NotImplementedError

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def logit(p: ArrayLike) -> NDArray[np.float64]:
    """Return log(p / (1 - p)) elementwise, accurate for every p strictly inside (0, 1).

    Parameters
    ----------
    p : array-like of finite values, each strictly between 0 and 1.

    Returns
    -------
    float64 array with the shape of p (0-d for a scalar). logit(1/2) is exactly 0, and the
    function is odd about 1/2: logit(1 - p) = -logit(p).

    Raises
    ------
    ValueError
        If any entry is NaN, infinite, or not strictly between 0 and 1.
    """
    raise NotImplementedError

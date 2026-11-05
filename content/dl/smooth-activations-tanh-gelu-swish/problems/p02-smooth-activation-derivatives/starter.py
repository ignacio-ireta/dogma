from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def smooth_derivatives(
    x: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return derivatives of tanh, exact GELU and beta-1 Swish as float64 arrays.

    Same-shape real finite x in [-1000, 1000]; scalar/empty arrays accepted.
    Raise ValueError for complex, nonfinite or out-of-range values. No mutation."""
    raise NotImplementedError

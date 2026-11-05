from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def smooth_activations(
    x: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return tanh, exact GELU and beta-1 Swish as same-shape float64 arrays.

    x is real finite array-like in [-1000, 1000], including scalar/empty arrays.
    Raise ValueError for complex, nonfinite or out-of-range inputs; do not mutate x."""
    raise NotImplementedError

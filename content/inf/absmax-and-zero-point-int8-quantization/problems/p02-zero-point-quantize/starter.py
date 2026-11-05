from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def zero_point_quantize(x: NDArray[np.float64]) -> tuple[NDArray[np.int8], float, int]:
    """Quantize x to int8 codes [-128, 127] with an asymmetric scale and zero point.

    The range is lo = min(min(x), 0) to hi = max(max(x), 0). scale = (hi - lo) / 255, or 1.0
    when lo == hi. zero_point = -128 - round_half_to_even(lo / scale) is the code of 0.0.
    Codes are clip(round_half_to_even(x / scale) + zero_point, -128, 127).
    Returns (q int8 with x.shape, scale float, zero_point int). Raise ValueError for bool,
    complex or object dtypes, ndim 0, empty or nonfinite input. Never mutate x.
    """
    raise NotImplementedError

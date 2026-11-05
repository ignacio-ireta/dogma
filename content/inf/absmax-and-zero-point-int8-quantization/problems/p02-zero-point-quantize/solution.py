from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _real_values(x: object) -> NDArray[np.float64]:
    """Validate the dtype before casting, so complex or bool data cannot pass silently."""
    raw = np.asarray(x)
    if raw.dtype.kind not in "iuf":
        raise ValueError("expected a real integer or float array, not bool, complex or object")
    if raw.ndim < 1:
        raise ValueError("expected an array with at least one axis")
    if raw.size == 0:
        raise ValueError("expected a nonempty array")
    values = raw.astype(np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("values must be finite")
    return values


def zero_point_quantize(x: NDArray[np.float64]) -> tuple[NDArray[np.int8], float, int]:
    """Quantize x to int8 codes with an asymmetric scale and zero point (see problem.md)."""
    values = _real_values(x)

    # The range must contain 0.0, so zero is exactly representable. Without this step an
    # all-positive tensor would waste codes below its minimum, and 0.0 might not land on a code.
    lo = min(float(values.min()), 0.0)
    hi = max(float(values.max()), 0.0)

    # 256 codes span 255 steps between lo and hi. An all-zero tensor has no width, so 1.0 is used.
    scale = 1.0 if hi == lo else (hi - lo) / 255.0
    if hi > lo and scale < np.finfo(np.float64).tiny:
        raise ValueError("scale underflows: (hi - lo) / 255 is not a normal float64")

    # Shift the grid so that lo lands on code -128. The rounded value keeps zero_point an integer.
    zero_point = int(-128 - np.round(lo / scale))

    # Round first, then add the integer zero point. The clip is needed at the top edge: when
    # lo / scale is a half-integer, hi can round one step past the grid. For example, lo = -253.5
    # and hi = 1.5 (scale 1) give hi the code 128 before clipping.
    codes = np.clip(np.round(values / scale) + zero_point, -128, 127).astype(np.int8)
    return codes, float(scale), zero_point

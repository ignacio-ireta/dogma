from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def zero_point_quantize(x: NDArray[np.float64]) -> tuple[NDArray[np.int8], float, int]:
    if not isinstance(x, np.ndarray):
        raise ValueError("x must be a numpy array")
    if x.dtype.kind not in "iuf":
        raise ValueError("invalid dtype")
    if x.ndim < 1 or x.size == 0:
        raise ValueError("x must be non-empty with ndim >= 1")
    xf = x.astype(np.float64)
    if not np.all(np.isfinite(xf)):
        raise ValueError("nonfinite value")
    lo = min(float(xf.min()), 0.0)
    hi = max(float(xf.max()), 0.0)
    if lo == hi:
        scale = 1.0
    else:
        span = hi - lo
        scale = span / 255.0 if np.isfinite(span) else hi / 255.0 - lo / 255.0
        if scale < 2.2250738585072014e-308:
            raise ValueError("scale underflows")
    zp = -128 - int(np.round(lo / scale))
    q = np.clip(np.round(xf / scale) + zp, -128, 127).astype(np.int8)
    return q, float(scale), int(zp)

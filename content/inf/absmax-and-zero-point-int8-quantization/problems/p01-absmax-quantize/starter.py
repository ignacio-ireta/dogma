from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def absmax_quantize(
    x: NDArray[np.float64], axis: int | None = None
) -> tuple[NDArray[np.int8], NDArray[np.float64]]:
    """Quantize x to int8 with symmetric absmax scales (scale = absmax / 127).

    axis=None uses one scale for the whole tensor (scale shape ()). An int axis names the
    channel axis and gives one scale per index along it (scale shape (x.shape[axis],)).
    A group whose absmax is 0 gets scale 1.0. Codes are round-half-to-even of x / scale,
    clipped to [-127, 127]. Returns (q int8 with x.shape, scale float64).
    Raise ValueError for bool/complex/object dtypes, ndim 0, empty or nonfinite input,
    or an invalid axis. Never mutate x.
    """
    raise NotImplementedError


def absmax_dequantize(
    q: NDArray[np.int8], scale: NDArray[np.float64], axis: int | None = None
) -> NDArray[np.float64]:
    """Return q * scale as float64, with scale broadcast along the channel axis.

    q must be int8. scale must be finite, positive float64 with shape () for axis=None or
    (q.shape[axis],) otherwise. Mismatches raise ValueError.
    """
    raise NotImplementedError

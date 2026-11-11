from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# Symmetric int8 grid: codes -127..127. The code -128 is left out so that the grid is symmetric
# around zero and the scale is absmax / 127 for both signs.
_QMAX = 127


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


def _normalize_axis(axis: object, ndim: int) -> int | None:
    """Map None through, and a (possibly negative) integer axis into [0, ndim)."""
    if axis is None:
        return None
    if isinstance(axis, bool) or not isinstance(axis, int | np.integer):
        raise ValueError("axis must be None or an integer")
    if not -ndim <= axis < ndim:
        raise ValueError("axis is out of range for this array")
    return int(axis) % ndim


def absmax_quantize(
    x: NDArray[np.float64], axis: int | None = None
) -> tuple[NDArray[np.int8], NDArray[np.float64]]:
    """Wrong on purpose: exact halves are rounded up, away from zero on the positive side."""
    values = _real_values(x)
    axis = _normalize_axis(axis, values.ndim)

    # Reduce over every axis except the channel axis. keepdims keeps the shape broadcastable
    # against x, so one division scales each group.
    if axis is None:
        reduce_axes: tuple[int, ...] | None = None
    else:
        reduce_axes = tuple(i for i in range(values.ndim) if i != axis)
    absmax = np.max(np.abs(values), axis=reduce_axes, keepdims=True)

    # scale = absmax / 127. An all-zero group gets scale 1.0, so no division by zero happens and
    # its codes are 0 exactly.
    scale = np.where(absmax > 0, absmax / _QMAX, 1.0)
    # A scale below the smallest normal float64 has too few bits to divide by, and may
    # underflow to 0.
    if np.any((absmax > 0) & (scale < np.finfo(np.float64).tiny)):
        raise ValueError("scale underflows: absmax / 127 is not a normal float64")

    # np.round rounds half to even, as problem.md requires. For a normal scale, |x| <= absmax
    # keeps x / scale within [-127, 127] up to float error, so the clip changes no code here.
    codes = np.clip(np.floor(values / scale + 0.5), -_QMAX, _QMAX).astype(np.int8)

    if axis is None:
        return codes, scale.reshape(())
    return codes, scale.reshape(values.shape[axis])


def absmax_dequantize(
    q: NDArray[np.int8], scale: NDArray[np.float64], axis: int | None = None
) -> NDArray[np.float64]:
    """Return q * scale as float64, with scale broadcast along the channel axis."""
    codes = np.asarray(q)
    if codes.dtype != np.int8:
        raise ValueError("q must be an int8 array")
    axis = _normalize_axis(axis, codes.ndim)

    s = np.asarray(scale)
    if s.dtype.kind != "f" or not np.all(np.isfinite(s)) or np.any(s <= 0):
        raise ValueError("scale must be finite, positive float64")
    if axis is None:
        if s.shape != ():
            raise ValueError("per-tensor scale must have shape ()")
        return codes.astype(np.float64) * float(s)
    if s.shape != (codes.shape[axis],):
        raise ValueError("per-channel scale must have shape (q.shape[axis],)")
    # Put the scale along the channel axis only, so it broadcasts to every row of that channel.
    shape = [1] * codes.ndim
    shape[axis] = -1
    return codes.astype(np.float64) * s.reshape(shape)

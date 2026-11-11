from __future__ import annotations

import numpy as np

_TINY = np.finfo(np.float64).tiny


def _values(x) -> np.ndarray:
    arr = np.asarray(x)
    if arr.dtype.kind not in "iuf":
        raise ValueError("x must be a real integer or float array")
    if arr.ndim < 1:
        raise ValueError("x must have at least one axis")
    if arr.size == 0:
        raise ValueError("x must not be empty")
    values = arr.astype(np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("x must be finite")
    return values


def _axis(axis, ndim: int):
    if axis is None:
        return None
    if isinstance(axis, bool) or not isinstance(axis, (int, np.integer)):
        raise ValueError("axis must be None or an integer")
    if not -ndim <= axis < ndim:
        raise ValueError("axis is out of range")
    return int(axis) % ndim


def absmax_quantize(x, axis=None):
    values = _values(x)
    ax = _axis(axis, values.ndim)
    if ax is None:
        absmax = np.max(np.abs(values), keepdims=True)
    else:
        others = tuple(i for i in range(values.ndim) if i != ax)
        absmax = np.max(np.abs(values), axis=others, keepdims=True)
    zero = absmax == 0
    scale = np.where(zero, 1.0, absmax / 127.0)
    if np.any(~zero & (scale < _TINY)):
        raise ValueError("scale underflows: absmax / 127 is not a normal float64")
    q = np.clip(np.round(values / scale), -127, 127).astype(np.int8)
    if ax is None:
        out_scale = np.asarray(scale.reshape(()), dtype=np.float64)
    else:
        out_scale = np.asarray(scale.reshape(values.shape[ax]), dtype=np.float64)
    return q, out_scale


def absmax_dequantize(q, scale, axis=None):
    q_arr = np.asarray(q)
    if q_arr.dtype != np.int8:
        raise ValueError("q must be an int8 array")
    s = np.asarray(scale)
    if s.dtype != np.float64:
        raise ValueError("scale must be a float64 array")
    ax = _axis(axis, q_arr.ndim)
    expected = () if ax is None else (q_arr.shape[ax],)
    if s.shape != expected:
        raise ValueError("scale shape must match the axis")
    if not np.all(np.isfinite(s)) or not np.all(s > 0):
        raise ValueError("scale must be finite and strictly positive")
    if ax is None:
        broadcast = s.reshape(())
    else:
        shape = [1] * q_arr.ndim
        shape[ax] = q_arr.shape[ax]
        broadcast = s.reshape(shape)
    return q_arr.astype(np.float64) * broadcast

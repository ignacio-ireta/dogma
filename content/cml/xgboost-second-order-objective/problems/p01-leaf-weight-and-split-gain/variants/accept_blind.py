from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike


def _side(g: ArrayLike, h: ArrayLike, name: str) -> tuple[float, float]:
    g_arr = np.array(g, dtype=float, copy=True)
    h_arr = np.array(h, dtype=float, copy=True)
    if g_arr.ndim != 1 or h_arr.ndim != 1:
        raise ValueError(f"{name}: gradients and hessians must be 1-D")
    if g_arr.shape != h_arr.shape:
        raise ValueError(f"{name}: gradient and hessian lengths differ")
    if not np.all(np.isfinite(g_arr)):
        raise ValueError(f"{name}: gradients must be finite")
    if not np.all(np.isfinite(h_arr)):
        raise ValueError(f"{name}: hessians must be finite")
    if np.any(h_arr < 0):
        raise ValueError(f"{name}: hessians must be zero or positive")
    return math.fsum(g_arr.tolist()), math.fsum(h_arr.tolist())


def leaf_weight_and_split_gain(
    g_left: ArrayLike,
    h_left: ArrayLike,
    g_right: ArrayLike,
    h_right: ArrayLike,
    *,
    reg_lambda: float,
    gamma: float,
) -> tuple[float, float, float]:
    """Return (w_left, w_right, gain) for one candidate split."""
    lam = float(reg_lambda)
    gam = float(gamma)
    if not math.isfinite(lam) or lam < 0:
        raise ValueError("reg_lambda must be a finite number that is zero or positive")
    if not math.isfinite(gam) or gam < 0:
        raise ValueError("gamma must be a finite number that is zero or positive")

    gl, hl = _side(g_left, h_left, "left")
    gr, hr = _side(g_right, h_right, "right")

    if hl + lam <= 0:
        raise ValueError("left side needs H + reg_lambda > 0")
    if hr + lam <= 0:
        raise ValueError("right side needs H + reg_lambda > 0")

    denom_l = hl + lam
    denom_r = hr + lam
    denom_p = hl + hr + lam

    w_left = -gl / denom_l
    w_right = -gr / denom_r
    gain = 0.5 * (gl * gl / denom_l + gr * gr / denom_r - (gl + gr) * (gl + gr) / denom_p) - gam
    return float(w_left), float(w_right), float(gain)

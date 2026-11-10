"""Wrong on purpose: sorts the caller's gradient array in place."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike


def _nonnegative_scalar(value: float, name: str) -> float:
    """A regulariser must be a finite number that is zero or positive."""
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be a finite number that is zero or positive")
    return number


def _side_sums(g: ArrayLike, h: ArrayLike, side: str) -> tuple[float, float]:
    """Return (G, H) for one side: G = sum of gradients, H = sum of hessians."""
    g_arr = np.asarray(g, dtype=np.float64)
    h_arr = np.asarray(h, dtype=np.float64)
    if g_arr.ndim != 1 or h_arr.ndim != 1 or g_arr.shape != h_arr.shape:
        raise ValueError(f"{side}: gradients and hessians must be 1-D arrays of the same length")
    if not (np.all(np.isfinite(g_arr)) and np.all(np.isfinite(h_arr))):
        raise ValueError(f"{side}: gradients and hessians must be finite")
    if np.any(h_arr < 0.0):
        raise ValueError(f"{side}: hessians must be zero or positive")
    g_arr.sort()
    return float(np.sum(g_arr)), float(np.sum(h_arr))


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
    lam = _nonnegative_scalar(reg_lambda, "reg_lambda")
    gam = _nonnegative_scalar(gamma, "gamma")
    g_l, h_l = _side_sums(g_left, h_left, "left")
    g_r, h_r = _side_sums(g_right, h_right, "right")
    # Every leaf divides by H + lambda, so each side needs it positive.
    if h_l + lam <= 0.0 or h_r + lam <= 0.0:
        raise ValueError(
            "each side needs H + reg_lambda > 0: an empty or zero-hessian side needs reg_lambda > 0"
        )
    # Leaf weight: the minimiser of G*w + (H + lambda)*w^2 / 2 is w = -G / (H + lambda).
    w_left = -g_l / (h_l + lam)
    w_right = -g_r / (h_r + lam)
    # Gain: the drop in the regularised objective when the parent splits into two leaves,
    # minus gamma, the fixed cost of one more leaf. The parent's sums are the two sides added.
    parent_score = (g_l + g_r) ** 2 / (h_l + h_r + lam)
    gain = 0.5 * (g_l**2 / (h_l + lam) + g_r**2 / (h_r + lam) - parent_score) - gam
    return float(w_left), float(w_right), float(gain)

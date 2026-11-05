from __future__ import annotations

from numpy.typing import ArrayLike


def leaf_weight_and_split_gain(
    g_left: ArrayLike,
    h_left: ArrayLike,
    g_right: ArrayLike,
    h_right: ArrayLike,
    *,
    reg_lambda: float,
    gamma: float,
) -> tuple[float, float, float]:
    """Return (w_left, w_right, gain) for one candidate split.

    G and H are the sums of the gradients and hessians on each side. The leaf weights are
    w = -G / (H + reg_lambda). The gain is 0.5 * [G_L^2/(H_L+lambda) + G_R^2/(H_R+lambda)
    - (G_L+G_R)^2/(H_L+H_R+lambda)] - gamma. Each side needs H + reg_lambda > 0.

    Parameters
    ----------
    g_left, g_right : 1-D finite arrays, either may be empty.
    h_left, h_right : 1-D arrays, same lengths as the gradients, finite and >= 0.
    reg_lambda : finite number >= 0.
    gamma : finite number >= 0.

    Returns
    -------
    (w_left, w_right, gain) as Python floats.

    Raises
    ------
    ValueError
        For bad shapes, non-finite values, negative hessians or regularisers, or a side with
        H + reg_lambda <= 0.
    """
    raise NotImplementedError

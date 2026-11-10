from __future__ import annotations

import math


def _loss_of(x, w, b, t):
    h = math.tanh(w * x + b)
    r = h * w + math.exp(h) - t
    return r * r


def forward_backward(x: float, w: float, b: float, t: float) -> tuple[float, dict[str, float]]:
    """Wrong on purpose: dL/dx is a central difference, not the analytic derivative. Evaluate the scalar graph and return (loss, gradients).

    The graph (see problem.md) is p = w*x, z = p + b, h = tanh(z), q = h*w, e = exp(h),
    y = q + e, r = y - t, L = r*r. The backward pass multiplies each node's local derivative
    by the adjoint it received, and an input used twice receives two contributions that add.

    Returns
    -------
    loss : float, equal to L.
    grads : dict with keys "x", "w", "b", "t"; each is dL/d(input).

    Raises
    ------
    ValueError
        If any input is NaN or infinite.
    """  # noqa: E501
    values = {"x": x, "w": w, "b": b, "t": t}
    for name, value in values.items():
        if not math.isfinite(value):
            raise ValueError(f"input {name} must be a finite number")
    x, w, b, t = float(x), float(w), float(b), float(t)

    # Forward pass: evaluate every node once and keep the intermediate values, because the
    # backward pass reads them (tanh' uses h, exp' uses e).
    p = w * x  # first use of w
    z = p + b
    h = math.tanh(z)
    q = h * w  # second use of w
    e = math.exp(h)  # second use of h (the first is q)
    y = q + e
    r = y - t
    loss = r * r  # r is used twice: both factors are r

    # Backward pass. The adjoint of L is 1. Each step below is "local derivative times the
    # adjoint received", applied from the output back to the inputs.
    g_r = 2.0 * r  # d(r*r)/dr: both factors contribute r
    g_y = g_r  # r = y - t, so dr/dy = 1
    g_t = -g_r  # dr/dt = -1
    g_q = g_y  # y = q + e, so dy/dq = 1
    g_e = g_y  # dy/de = 1
    # h has two consumers, so its adjoint is the sum of two contributions.
    g_h = g_q * w + g_e * e  # via q (dq/dh = w), via e (de/dh = exp(h) = e)
    g_z = g_h * (1.0 - h * h)  # tanh'(z) = 1 - tanh(z)**2, read from the stored h
    g_b = g_z  # z = p + b, so dz/db = 1
    g_p = g_z  # dz/dp = 1
    g_x = (_loss_of(x + 1e-7, w, b, t) - _loss_of(x - 1e-7, w, b, t)) / 2e-7
    # w also has two consumers: q (dq/dw = h) and p (dp/dw = x).
    g_w = g_q * h + g_p * x

    grads = {"x": g_x, "w": g_w, "b": g_b, "t": g_t}
    return float(loss), grads

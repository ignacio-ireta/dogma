from __future__ import annotations

import math


def _check(name: str, value: float) -> float:
    v = float(value)
    if math.isnan(v) or math.isinf(v):
        raise ValueError(f"input {name} must be a finite number")
    return v


def forward_backward(x: float, w: float, b: float, t: float) -> tuple[float, dict[str, float]]:
    """Evaluate the scalar graph L = (y - t)**2 and return (loss, gradients).

    Forward: p = w*x, z = p + b, h = tanh(z), q = h*w, e = exp(h), y = q + e, r = y - t, L = r*r.
    Backward: dL/dx, dL/dw, dL/db, dL/dt. A value used twice (w, h, r) sums both contributions.

    Returns (loss, {"x": dL/dx, "w": dL/dw, "b": dL/db, "t": dL/dt}), all floats.
    Raises ValueError when any input is NaN or infinite.
    """
    x = _check("x", x)
    w = _check("w", w)
    b = _check("b", b)
    t = _check("t", t)

    # Forward pass.
    p = w * x
    z = p + b
    h = math.tanh(z)
    q = h * w
    e = math.exp(h)
    y = q + e
    r = y - t
    loss = r * r

    # Backward pass.
    dL_dr = 2.0 * r
    dL_dt = -dL_dr
    dL_dy = dL_dr
    dL_dq = dL_dy
    dL_de = dL_dy
    # h feeds q (through h*w) and e (through exp).
    dL_dh = dL_dq * w + dL_de * math.exp(h)
    dL_dz = dL_dh * (1.0 - h * h)
    dL_db = dL_dz
    dL_dp = dL_dz
    dL_dx = dL_dp * w
    # w feeds p (through w*x) and q (through h*w).
    dL_dw = dL_dp * x + dL_dq * h

    grads = {
        "x": float(dL_dx),
        "w": float(dL_dw),
        "b": float(dL_db),
        "t": float(dL_dt),
    }
    return float(loss), grads

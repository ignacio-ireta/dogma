from __future__ import annotations


def forward_backward(x: float, w: float, b: float, t: float) -> tuple[float, dict[str, float]]:
    """Evaluate the scalar graph L = (y - t)**2 and return (loss, gradients).

    Forward: p = w*x, z = p + b, h = tanh(z), q = h*w, e = exp(h), y = q + e, r = y - t, L = r*r.
    Backward: dL/dx, dL/dw, dL/db, dL/dt. A value used twice (w, h, r) sums both contributions.

    Returns (loss, {"x": dL/dx, "w": dL/dw, "b": dL/db, "t": dL/dt}), all floats.
    Raises ValueError when any input is NaN or infinite.
    """
    raise NotImplementedError

from __future__ import annotations

from collections.abc import Callable


def numerical_derivative(
    f: Callable[[float], float],
    x: float,
    method: str = "central",
    h: float | None = None,
) -> float:
    """Estimate f'(x) with a forward or central finite difference.

    Parameters
    ----------
    f : scalar function of one float; it is only evaluated, never modified.
    x : finite real number.
    method : "forward" (error O(h)) or "central" (error O(h^2)); anything else raises.
    h : positive finite step. None selects a default scaled to max(1, |x|):
        sqrt(eps) * max(1, |x|) for forward, cbrt(eps) * max(1, |x|) for central.

    Returns
    -------
    float : the estimate of f'(x).

    Raises
    ------
    ValueError
        If method is unknown, x is not finite, or h is not a positive finite number.
    """
    raise NotImplementedError

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

_TOL = 1e-9


def marginals_and_conditional(
    joint: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Wrong on purpose: Return (p_x, p_y, p_y_given_x) for a joint probability table P[X=i, Y=j].

    Marginals sum out one axis each. The conditional P(Y=j | X=i) divides each row by its
    marginal P(X=i). A row with P(X=i) = 0 has no defined conditional, so it is NaN.
    """
    # Copy into float64: integer tables must work, and the caller's array is never written.
    P = np.array(joint, dtype=np.float64)

    if P.ndim != 2:
        raise ValueError(f"joint must be a 2-D table, got {P.ndim}-D")
    # A NaN makes every comparison False, so "P < 0" alone would let it through. Check finiteness.
    if not np.isfinite(P).all():
        raise ValueError("joint entries must be finite numbers (no NaN or inf)")
    if (P < 0).any():
        raise ValueError("joint entries must be non-negative probabilities")
    total = P.sum()
    if abs(total - 1.0) > _TOL:
        raise ValueError(f"joint must sum to 1 within {_TOL}, got {total!r}")

    # Summing out an axis gives a marginal: rows for P(X), columns for P(Y).
    p_x = P.sum(axis=1)  # shape (m,)
    p_y = P.sum(axis=0)  # shape (k,)

    # P(Y=j | X=i) = P(X=i, Y=j) / P(X=i). Division by zero is undefined, so those rows stay NaN.
    # A tiny but positive P(X=i) still defines the row: test > 0, not > some tolerance.
    p_y_given_x = np.zeros(P.shape)
    defined = p_x > 0
    p_y_given_x[defined] = P[defined] / p_x[defined, None]
    return p_x, p_y, p_y_given_x

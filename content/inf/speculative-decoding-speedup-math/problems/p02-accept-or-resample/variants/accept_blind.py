from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def _check_dist(name: str, values: Any) -> NDArray[np.float64]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or arr.size < 1:
        raise ValueError(f"{name} must be a non-empty 1-D array")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    if np.any(arr < 0):
        raise ValueError(f"{name} must be nonnegative")
    if abs(float(arr.sum()) - 1.0) > 1e-9:
        raise ValueError(f"{name} must sum to 1 within 1e-9")
    return arr.copy()


def _first_index_above(weights: NDArray[np.float64], u: float) -> int:
    total = float(weights.sum())
    cdf = np.cumsum(weights / total)
    cdf[-1] = 1.0
    return int(np.searchsorted(cdf, u, side="right"))


def accept_or_resample(
    p: NDArray[np.float64],
    q: NDArray[np.float64],
    draft_token: int,
    rng: Any,
) -> int:
    """Return the token one speculative step emits at a single position.

    p: target probabilities, 1-D, length V >= 1, finite, nonnegative, sum 1 within 1e-9.
    q: draft probabilities with the same rules. draft_token: int in [0, V) with q[draft_token] > 0.
    rng: object with random() returning floats in [0, 1). Exactly one draw when accepted, two
    when rejected (accept uniform, then resample uniform). Returns a Python int in [0, V).
    Raise ValueError for invalid arguments, including a zero draft probability at draft_token.
    """
    p_arr = _check_dist("p", p)
    q_arr = _check_dist("q", q)
    if p_arr.shape != q_arr.shape:
        raise ValueError("p and q must have the same length")

    if isinstance(draft_token, (bool, np.bool_)) or not isinstance(draft_token, (int, np.integer)):
        raise ValueError("draft_token must be an integer")
    x = int(draft_token)
    vocab = p_arr.shape[0]
    if not 0 <= x < vocab:
        raise ValueError("draft_token out of range")
    if q_arr[x] == 0.0:
        raise ValueError("draft_token has zero draft probability")

    u1 = float(rng.random())
    ratio = min(1.0, float(p_arr[x]) / float(q_arr[x]))
    if u1 < ratio:
        return x

    residual = np.maximum(0.0, p_arr - q_arr)
    if float(residual.sum()) == 0.0:
        residual = p_arr
    u2 = float(rng.random())
    return _first_index_above(residual, u2)

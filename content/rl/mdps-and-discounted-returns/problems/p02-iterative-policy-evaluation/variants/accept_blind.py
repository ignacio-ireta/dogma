from __future__ import annotations

import math
import numbers

import numpy as np
from numpy.typing import NDArray


def _real_array(name: str, x: object, ndim: int) -> NDArray[np.float64]:
    if not isinstance(x, np.ndarray):
        raise ValueError(f"{name} must be a NumPy array")
    if x.dtype.kind not in ("i", "u", "f"):
        raise ValueError(f"{name} must be a real numeric array")
    if x.ndim != ndim:
        raise ValueError(f"{name} has the wrong number of dimensions")
    arr = x.astype(np.float64, copy=True)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


def _real_scalar(name: str, x: object) -> float:
    if isinstance(x, (bool, np.bool_)) or not isinstance(x, numbers.Real):
        raise ValueError(f"{name} must be a real number")
    value = float(x)
    if math.isnan(value) or math.isinf(value):
        raise ValueError(f"{name} must be finite")
    return value


def iterative_policy_evaluation(
    P: NDArray[np.float64],
    R: NDArray[np.float64],
    pi: NDArray[np.float64],
    gamma: float,
    tol: float = 1e-10,
) -> NDArray[np.float64]:
    """Return V^pi as a float64 array of shape (S,) by iterating the Bellman expectation update.

    P: (S, A, S) transition probabilities, rows over s' sum to 1.
    R: (S, A) expected rewards. pi: (S, A) policy, rows over a sum to 1.
    gamma: real number in [0, 1). tol: positive stopping threshold on the max-norm change.
    Raises ValueError on invalid input. Never mutates inputs; never calls numpy.linalg.
    """
    P_arr = _real_array("P", P, 3)
    R_arr = _real_array("R", R, 2)
    pi_arr = _real_array("pi", pi, 2)

    S, A, S2 = P_arr.shape
    if S2 != S:
        raise ValueError("P must have shape (S, A, S)")
    if R_arr.shape != (S, A):
        raise ValueError("R must have shape (S, A)")
    if pi_arr.shape != (S, A):
        raise ValueError("pi must have shape (S, A)")
    if S == 0 or A == 0:
        raise ValueError("P must have at least one state and one action")
    if np.any(P_arr < 0):
        raise ValueError("P must be non-negative")
    if np.any(pi_arr < 0):
        raise ValueError("pi must be non-negative")
    if not np.allclose(P_arr.sum(axis=2), 1.0, rtol=0.0, atol=1e-9):
        raise ValueError("each P[s, a, :] must sum to 1")
    if not np.allclose(pi_arr.sum(axis=1), 1.0, rtol=0.0, atol=1e-9):
        raise ValueError("each pi[s, :] must sum to 1")

    g = _real_scalar("gamma", gamma)
    if not (0.0 <= g < 1.0):
        raise ValueError("gamma must lie in [0, 1)")
    threshold = _real_scalar("tol", tol)
    if threshold <= 0.0:
        raise ValueError("tol must be positive")

    V = np.zeros(S, dtype=np.float64)
    while True:
        Q = R_arr + g * np.einsum("sap,p->sa", P_arr, V)
        V_new = np.sum(pi_arr * Q, axis=1)
        delta = float(np.max(np.abs(V_new - V)))
        V = V_new
        if delta < threshold:
            break
    return V

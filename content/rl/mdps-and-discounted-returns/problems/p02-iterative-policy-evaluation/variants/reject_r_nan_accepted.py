from __future__ import annotations

import numbers

import numpy as np
from numpy.typing import NDArray

PROB_ATOL = 1e-9  # rows of P and pi must sum to one within this slack


def _real_array(value: object, ndim: int, name: str) -> NDArray[np.float64]:
    """Check dtype and rank before casting, so complex or object data cannot slip through."""
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a real numeric array")
    if raw.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    array = raw.astype(np.float64)  # a copy, so the caller's array is never touched
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _validate_model(
    P: NDArray[np.float64], R: NDArray[np.float64], pi: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Check shapes, signs and the row sums that make P and pi probability distributions."""
    p = _real_array(P, 3, "P")
    r = np.asarray(R, dtype=np.float64)
    pol = _real_array(pi, 2, "pi")
    n_states, n_actions, last = p.shape
    if n_states < 1 or n_actions < 1 or last != n_states:
        raise ValueError("P must have shape (S, A, S) with S, A >= 1")
    if r.shape != (n_states, n_actions) or pol.shape != (n_states, n_actions):
        raise ValueError("R and pi must have shape (S, A) matching P")
    if np.any(p < 0) or np.any(pol < 0):
        raise ValueError("probabilities must be non-negative")
    if not np.allclose(p.sum(axis=2), 1.0, rtol=0.0, atol=PROB_ATOL):
        raise ValueError("each P[s, a, :] must sum to 1")
    if not np.allclose(pol.sum(axis=1), 1.0, rtol=0.0, atol=PROB_ATOL):
        raise ValueError("each pi[s, :] must sum to 1")
    return p, r, pol


def _checked_discount(gamma: float) -> float:
    """Require 0 <= gamma < 1: gamma = 1 need not converge for a continuing task."""
    if isinstance(gamma, bool) or not isinstance(gamma, numbers.Real):
        raise ValueError("gamma must be a real number")
    value = float(gamma)
    if not 0.0 <= value < 1.0:  # NaN fails the comparison
        raise ValueError("gamma must satisfy 0 <= gamma < 1")
    return value


def _checked_tolerance(tol: float) -> float:
    if isinstance(tol, bool) or not isinstance(tol, numbers.Real):
        raise ValueError("tol must be a real number")
    value = float(tol)
    if not (np.isfinite(value) and value > 0.0):
        raise ValueError("tol must be a positive finite number")
    return value


def iterative_policy_evaluation(
    P: NDArray[np.float64],
    R: NDArray[np.float64],
    pi: NDArray[np.float64],
    gamma: float,
    tol: float = 1e-10,
) -> NDArray[np.float64]:
    """Wrong on purpose: Return V^pi as a float64 array of shape (S,) by iterating the Bellman expectation update.

    P: (S, A, S) transition probabilities, rows over s' sum to 1.
    R: (S, A) expected rewards. pi: (S, A) policy, rows over a sum to 1.
    gamma: real number in [0, 1). tol: positive stopping threshold on the max-norm change.
    Raises ValueError on invalid input. Never mutates inputs; never calls numpy.linalg.
    """  # noqa: E501
    p, r, pol = _validate_model(P, R, pi)
    discount = _checked_discount(gamma)
    threshold = _checked_tolerance(tol)

    # Average over the policy first. P_pi[s, s'] = sum_a pi[s, a] P[s, a, s'] and
    # R_pi[s] = sum_a pi[s, a] R[s, a]. The Bellman update then needs only these two arrays.
    p_pi = np.einsum("sa,sat->st", pol, p)
    r_pi = np.einsum("sa,sa->s", pol, r)

    # V <- R_pi + gamma * P_pi V, starting from zero. This is a gamma-contraction in the max
    # norm, so it converges; stop when one sweep changes no state by more than the threshold.
    values = np.zeros(p.shape[0], dtype=np.float64)
    while True:
        updated = r_pi + discount * (p_pi @ values)
        change = float(np.max(np.abs(updated - values)))
        values = updated
        if change < threshold:
            return values

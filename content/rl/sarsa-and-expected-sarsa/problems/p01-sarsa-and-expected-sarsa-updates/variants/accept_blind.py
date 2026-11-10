from __future__ import annotations

import numbers

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _is_index(value: object) -> bool:
    return isinstance(value, numbers.Integral) and not isinstance(value, (bool, np.bool_))


def _is_real(value: object) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, (bool, np.bool_))


def _check_table(Q: ArrayLike) -> NDArray[np.float64]:
    raw = np.asarray(Q)
    if raw.ndim != 2 or raw.shape[0] < 1 or raw.shape[1] < 1:
        raise ValueError("Q must be a non-empty 2-D array")
    if not (np.issubdtype(raw.dtype, np.integer) or np.issubdtype(raw.dtype, np.floating)):
        raise ValueError("Q must be real-valued")
    table = np.array(raw, dtype=np.float64)
    if not np.all(np.isfinite(table)):
        raise ValueError("Q entries must be finite")
    return table


def _check_index(value: object, size: int, name: str) -> int:
    if not _is_index(value) or not 0 <= int(value) < size:
        raise ValueError(f"{name} index is out of range")
    return int(value)


def _check_params(alpha: object, gamma: object, epsilon: object = 0.0) -> None:
    for name, value in (("alpha", alpha), ("gamma", gamma), ("epsilon", epsilon)):
        if not _is_real(value) or not np.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must lie in [0, 1]")


def _check_reward(r: object) -> float:
    if not _is_real(r) or not np.isfinite(float(r)):
        raise ValueError("reward must be finite")
    return float(r)


def sarsa_update(  # noqa: PLR0917 (the contract fixes eight arguments)
    Q: ArrayLike,
    s: int,
    a: int,
    r: float,
    s_next: int | None,
    a_next: int | None,
    alpha: float,
    gamma: float,
) -> NDArray[np.float64]:
    table = _check_table(Q)
    n_states, n_actions = table.shape
    si = _check_index(s, n_states, "state")
    ai = _check_index(a, n_actions, "action")
    reward = _check_reward(r)
    _check_params(alpha, gamma)

    if s_next is None:
        target = reward
    else:
        ns = _check_index(s_next, n_states, "next state")
        na = _check_index(a_next, n_actions, "next action")
        target = reward + float(gamma) * table[ns, na]

    out = table.copy()
    out[si, ai] = table[si, ai] + float(alpha) * (target - table[si, ai])
    return out


def expected_sarsa_update(  # noqa: PLR0917 (the contract fixes eight arguments)
    Q: ArrayLike,
    s: int,
    a: int,
    r: float,
    s_next: int | None,
    alpha: float,
    gamma: float,
    epsilon: float,
) -> NDArray[np.float64]:
    table = _check_table(Q)
    n_states, n_actions = table.shape
    si = _check_index(s, n_states, "state")
    ai = _check_index(a, n_actions, "action")
    reward = _check_reward(r)
    _check_params(alpha, gamma, epsilon)
    eps = float(epsilon)

    if s_next is None:
        target = reward
    else:
        ns = _check_index(s_next, n_states, "next state")
        row = table[ns]
        tied = row == row.max()
        k = int(np.count_nonzero(tied))
        policy = np.full(n_actions, eps / n_actions)
        policy[tied] += (1.0 - eps) / k
        target = reward + float(gamma) * float(policy @ row)

    out = table.copy()
    out[si, ai] = table[si, ai] + float(alpha) * (target - table[si, ai])
    return out

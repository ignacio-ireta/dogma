from __future__ import annotations

import numbers

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _checked_table(Q: ArrayLike) -> NDArray[np.float64]:
    """Return Q as a float64 copy, after checking its shape and values."""
    table = np.asarray(Q)
    if table.ndim != 2 or table.shape[0] < 1 or table.shape[1] < 1:
        raise ValueError("Q must be a non-empty 2-D array of shape (S, A)")
    if table.dtype.kind not in "iuf":
        raise ValueError("Q must hold real numbers")
    values = table.astype(np.float64)  # a copy, so the caller's table is never written
    if not np.all(np.isfinite(values)):
        raise ValueError("Q must hold finite values")
    return values


def _checked_index(value: object, size: int, name: str) -> int:
    # bool is an int subclass in Python, so it is rejected explicitly.
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be an integer index")
    if not 0 <= int(value) < size:
        raise ValueError(f"{name} is out of range for a table of this size")
    return int(value)


def _checked_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a real number")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _checked_unit(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a real number in [0, 1]")
    number = float(value)
    if not 0.0 <= number <= 1.0:  # NaN fails this comparison too
        raise ValueError(f"{name} must lie in [0, 1]")
    return number


def _checked_step(
    table: NDArray[np.float64], s: object, a: object, r: object, s_next: object
) -> tuple[int, int, float, int | None]:
    """Validate the transition's indices and reward against the table's shape."""
    n_states, n_actions = table.shape
    state = _checked_index(s, n_states, "state s")
    action = _checked_index(a, n_actions, "action a")
    reward = _checked_real(r, "reward r")
    next_state = None if s_next is None else _checked_index(s_next, n_states, "next state")
    return state, action, reward, next_state


def _apply(
    table: NDArray[np.float64], s: int, a: int, alpha: float, target: float
) -> NDArray[np.float64]:
    """Move one entry toward the target: Q + alpha * (target - Q). Returns a new table."""
    updated = table.copy()
    updated[s, a] = table[s, a] + alpha * (target - table[s, a])
    return updated


def _epsilon_greedy_weights(row: NDArray[np.float64], epsilon: float) -> NDArray[np.float64]:
    """Policy pi(b) over one state's actions: epsilon/A everywhere, plus (1-epsilon)/k on ties."""
    n_actions = row.shape[0]
    greedy = row == row.max()  # exact equality defines a tie
    k = int(greedy.sum())
    weights = np.full(n_actions, epsilon / n_actions)
    # The exploit share is split evenly over the k tied actions. Tied actions have equal values,
    # so this split does not change the expected target, only which action the sampler favors.
    weights[greedy] += (1.0 - epsilon) / k
    return weights


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
    """Wrong on purpose: validates a_next even when the next state is terminal."""
    table = _checked_table(Q)
    state, action, reward, next_state = _checked_step(table, s, a, r, s_next)
    step = _checked_unit(alpha, "alpha")
    discount = _checked_unit(gamma, "gamma")
    next_action = _checked_index(a_next, table.shape[1], "next action a_next")
    if next_state is None:
        target = reward  # terminal: no bootstrap, and a_next is not read
    else:
        target = reward + discount * table[next_state, next_action]
    return _apply(table, state, action, step, target)


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
    """Return a new table with Q[s, a] moved toward the expected Sarsa target."""
    table = _checked_table(Q)
    state, action, reward, next_state = _checked_step(table, s, a, r, s_next)
    step = _checked_unit(alpha, "alpha")
    discount = _checked_unit(gamma, "gamma")
    rate = _checked_unit(epsilon, "epsilon")
    if next_state is None:
        target = reward
    else:
        # Average the next state's values under the policy, instead of sampling one action.
        weights = _epsilon_greedy_weights(table[next_state], rate)
        target = reward + discount * float(weights @ table[next_state])
    return _apply(table, state, action, step, target)

from __future__ import annotations

import numbers

import numpy as np
from numpy.typing import NDArray

ROWS = 3
COLS = 3
N_ACTIONS = 4
GOAL = 8  # bottom-right corner; entering it ends the episode with reward 1
START = 0  # top-left corner
MOVES = ((-1, 0), (0, 1), (1, 0), (0, -1))  # up, right, down, left as (row, col) steps


def _checked_unit(value: float, name: str, *, low_open: bool) -> float:
    """A real number in (0, 1] (low_open) or [0, 1]; bool and NaN are rejected."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a real number")
    x = float(value)
    ok = (0.0 < x <= 1.0) if low_open else (0.0 <= x <= 1.0)  # NaN fails both comparisons
    if not ok:
        raise ValueError(f"{name} is out of range")
    return x


def _checked_index(value: int, size: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be an integer index")
    if not 0 <= int(value) < size:
        raise ValueError(f"{name} is out of range")
    return int(value)


def q_learning_update(  # noqa: PLR0917 (the contract fixes these arguments)
    Q: NDArray[np.float64],
    s: int,
    a: int,
    r: float,
    s_next: int,
    done: bool,
    alpha: float,
    gamma: float,
) -> NDArray[np.float64]:
    """Return a new table after one Q-learning update; Q itself is not modified.

    Q: 2-D finite float array (S, A). s, a, s_next: integer indices in range. r: finite real.
    done: True when s_next is terminal; then the target is r with no bootstrap.
    alpha in (0, 1], gamma in [0, 1]. Raises ValueError on invalid input.
    """
    table = np.asarray(Q)
    if table.ndim != 2 or table.size == 0 or table.dtype.kind not in "iuf":
        raise ValueError("Q must be a non-empty two-dimensional real array")
    if not np.all(np.isfinite(table)):
        raise ValueError("Q must be finite")
    n_states, n_actions = table.shape
    state = _checked_index(s, n_states, "s")
    action = _checked_index(a, n_actions, "a")
    next_state = _checked_index(s_next, n_states, "s_next")
    if isinstance(r, bool) or not isinstance(r, numbers.Real) or not np.isfinite(float(r)):
        raise ValueError("r must be a finite real number")
    rate = _checked_unit(alpha, "alpha", low_open=True)
    discount = _checked_unit(gamma, "gamma", low_open=False)

    updated = table.astype(np.float64, copy=True)  # the caller's table is never modified
    # Target: r when the next state is terminal, else r + gamma * max_a' Q(s', a').
    # The max is over the next state's actions whatever the behaviour policy does next.
    bootstrap = 0.0 if done else discount * float(np.max(table[next_state]))
    target = float(r) + bootstrap
    updated[state, action] += rate * (target - updated[state, action])
    return updated


def _grid_step(state: int, action: int) -> tuple[int, float, bool]:
    """Deterministic gridworld move: off-grid moves stay put; entering GOAL pays 1 and ends."""
    row, col = divmod(state, COLS)
    d_row, d_col = MOVES[action]
    new_row, new_col = row + d_row, col + d_col
    if 0 <= new_row < ROWS and 0 <= new_col < COLS:
        next_state = new_row * COLS + new_col
    else:
        next_state = state
    reward = 1.0 if next_state == GOAL else 0.0
    return next_state, reward, next_state == GOAL


def train_q_learning(  # noqa: PLR0917 (the contract fixes these arguments)
    n_episodes: int,
    seed: int,
    alpha: float = 0.5,
    gamma: float = 0.9,
    epsilon: float = 0.3,
    max_steps: int = 100,
) -> NDArray[np.float64]:
    """Wrong on purpose: accepts max_steps equal to zero. Return the (9, 4) Q table after n_episodes of epsilon-greedy Q-learning on the gridworld.

    Each episode starts at state 0 and ends at the goal or after max_steps steps. Action choice
    follows the epsilon-greedy draw order of rl/epsilon-greedy-exploration: one rng.random() per
    choice; if it is below epsilon, one rng.integers(4) and that action; otherwise the lowest-index
    greedy action with no further draw. rng = numpy.random.default_rng(seed).
    """  # noqa: E501
    if isinstance(n_episodes, bool) or not isinstance(n_episodes, numbers.Integral):
        raise ValueError("n_episodes must be an integer")
    if n_episodes < 0:
        raise ValueError("n_episodes must be non-negative")
    if isinstance(seed, bool) or not isinstance(seed, numbers.Integral) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if isinstance(max_steps, bool) or not isinstance(max_steps, numbers.Integral) or max_steps < 0:
        raise ValueError("max_steps must be an integer >= 1")
    rate = _checked_unit(alpha, "alpha", low_open=True)
    discount = _checked_unit(gamma, "gamma", low_open=False)
    explore = _checked_unit(epsilon, "epsilon", low_open=False)

    rng = np.random.default_rng(seed)
    n_states = ROWS * COLS
    Q = np.zeros((n_states, N_ACTIONS), dtype=np.float64)
    for _ in range(int(n_episodes)):
        state = START
        for _ in range(int(max_steps)):
            # Epsilon-greedy choice with the fixed draw order from the contract.
            if rng.random() < explore:
                action = int(rng.integers(N_ACTIONS))
            else:
                action = int(np.argmax(Q[state]))
            next_state, reward, done = _grid_step(state, action)
            Q = q_learning_update(Q, state, action, reward, next_state, done, rate, discount)
            state = next_state
            if done:
                break
    return Q

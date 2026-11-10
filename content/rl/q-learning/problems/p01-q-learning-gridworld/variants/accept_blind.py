from __future__ import annotations

import math
import numbers

import numpy as np
from numpy.typing import NDArray

N_STATES = 9
N_ACTIONS = 4
GOAL = 8


def _is_int(x: object) -> bool:
    return isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))


def _real(name: str, x: object) -> float:
    if isinstance(x, (bool, np.bool_)) or not isinstance(x, numbers.Real):
        raise ValueError(f"{name} must be a real number")
    value = float(x)
    if math.isnan(value) or math.isinf(value):
        raise ValueError(f"{name} must be finite")
    return value


def _check_index(name: str, x: object, size: int) -> int:
    if not _is_int(x):
        raise ValueError(f"{name} must be an integer index")
    value = int(x)
    if not (0 <= value < size):
        raise ValueError(f"{name} is out of range")
    return value


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
    """Return a new table after one update Q[s,a] += alpha * (target - Q[s,a]).

    target = r if done, else r + gamma * max_a' Q[s_next, a']. Q is not modified.
    Q: 2-D finite real array (S, A). s, a, s_next: in-range integer indices. alpha in (0, 1],
    gamma in [0, 1]. Raises ValueError on invalid input.
    """
    if not isinstance(Q, np.ndarray) or Q.dtype.kind not in ("i", "u", "f"):
        raise ValueError("Q must be a real NumPy array")
    if Q.ndim != 2:
        raise ValueError("Q must be two-dimensional")
    table = Q.astype(np.float64, copy=True)
    if not np.all(np.isfinite(table)):
        raise ValueError("Q must be finite")
    n_states, n_actions = table.shape

    si = _check_index("s", s, n_states)
    ai = _check_index("a", a, n_actions)
    sn = _check_index("s_next", s_next, n_states)
    rv = _real("r", r)
    if not isinstance(done, (bool, np.bool_)):
        raise ValueError("done must be a bool")
    al = _real("alpha", alpha)
    if not (0.0 < al <= 1.0):
        raise ValueError("alpha must lie in (0, 1]")
    g = _real("gamma", gamma)
    if not (0.0 <= g <= 1.0):
        raise ValueError("gamma must lie in [0, 1]")

    if done:
        target = rv
    else:
        target = rv + g * float(np.max(table[sn]))
    table[si, ai] = table[si, ai] + al * (target - table[si, ai])
    return table


def train_q_learning(  # noqa: PLR0912, PLR0917
    n_episodes: int,
    seed: int,
    alpha: float = 0.5,
    gamma: float = 0.9,
    epsilon: float = 0.3,
    max_steps: int = 100,
) -> NDArray[np.float64]:
    """Return the (9, 4) Q table after n_episodes of epsilon-greedy Q-learning on the 3x3 gridworld.

    Episodes start at state 0 and end at the goal (state 8) or after max_steps steps. Action draws
    follow the contract in problem.md: rng.random() on every choice, then rng.integers(4) only when
    exploring, then the lowest-index greedy action. rng = numpy.random.default_rng(seed).
    Raises ValueError on invalid arguments.
    """
    if not _is_int(n_episodes) or n_episodes < 0:
        raise ValueError("n_episodes must be an integer >= 0")
    if not _is_int(seed) or seed < 0:
        raise ValueError("seed must be an integer >= 0")
    if not _is_int(max_steps) or max_steps < 1:
        raise ValueError("max_steps must be an integer >= 1")
    al = _real("alpha", alpha)
    if not (0.0 < al <= 1.0):
        raise ValueError("alpha must lie in (0, 1]")
    g = _real("gamma", gamma)
    if not (0.0 <= g <= 1.0):
        raise ValueError("gamma must lie in [0, 1]")
    eps = _real("epsilon", epsilon)
    if not (0.0 <= eps <= 1.0):
        raise ValueError("epsilon must lie in [0, 1]")

    rng = np.random.default_rng(int(seed))
    Q = np.zeros((N_STATES, N_ACTIONS), dtype=np.float64)
    for _ in range(int(n_episodes)):
        s = 0
        for _step in range(int(max_steps)):
            u = rng.random()
            if u < eps:
                a = int(rng.integers(N_ACTIONS))
            else:
                a = int(np.argmax(Q[s]))

            row, col = divmod(s, 3)
            if a == 0:
                row = max(row - 1, 0)
            elif a == 1:
                col = min(col + 1, 2)
            elif a == 2:
                row = min(row + 1, 2)
            else:
                col = max(col - 1, 0)
            s_next = 3 * row + col

            done = s_next == GOAL
            reward = 1.0 if done else 0.0
            Q = q_learning_update(Q, s, a, reward, s_next, done, al, g)
            s = s_next
            if done:
                break
    return Q

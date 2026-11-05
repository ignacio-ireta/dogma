from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


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
    raise NotImplementedError


def train_q_learning(  # noqa: PLR0917 (the contract fixes these arguments)
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
    raise NotImplementedError

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


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
    """Return a new table with Q[s, a] moved toward r + gamma * Q[s_next, a_next].

    Parameters
    ----------
    Q : 2-D real array-like of shape (S, A), finite entries. Never modified.
    s, a : integer state and action indices.
    r : finite reward.
    s_next : integer next state, or None when terminal. Then the target is r.
    a_next : integer action taken next in s_next. Ignored when s_next is None.
    alpha, gamma : real numbers in [0, 1].

    Returns
    -------
    (S, A) float64 array. Only entry (s, a) differs from Q.

    Raises
    ------
    ValueError
        For an invalid table, index, reward or parameter.
    """
    raise NotImplementedError


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
    """Return a new table with Q[s, a] moved toward the expected Sarsa target.

    The target averages Q[s_next, :] under the epsilon-greedy policy. Ties among the largest
    values split the exploit share 1 - epsilon evenly, and every action gets epsilon / A.
    Terminal s_next gives the target r. The input is never modified.

    Returns
    -------
    (S, A) float64 array. Only entry (s, a) differs from Q.

    Raises
    ------
    ValueError
        For an invalid table, index, reward or parameter.
    """
    raise NotImplementedError

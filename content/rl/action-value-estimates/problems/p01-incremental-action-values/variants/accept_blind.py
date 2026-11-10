from __future__ import annotations

import numbers

import numpy as np
from numpy.typing import NDArray


def _is_real_number(value: object) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, (bool, np.bool_))


def action_value_estimates(  # noqa: PLR0912
    actions: NDArray[np.int64],
    rewards: NDArray[np.float64],
    n_actions: int,
    step_size: float | None = None,
    initial_value: float = 0.0,
) -> NDArray[np.float64]:
    """Return the final action-value estimates, shape (n_actions,), after a stream of pulls.

    actions: 1-D integer indices in [0, n_actions). rewards: 1-D finite real array, same length.
    step_size None uses the sample average Q <- Q + (1/n)(r - Q), with n counting this pull.
    A number alpha in (0, 1] uses Q <- Q + alpha(r - Q). Untried actions keep initial_value.
    Raises ValueError on invalid input. Never mutates the input arrays.
    """
    if not isinstance(actions, np.ndarray) or not isinstance(rewards, np.ndarray):
        raise ValueError("actions and rewards must be NumPy arrays")
    if actions.ndim != 1 or rewards.ndim != 1:
        raise ValueError("actions and rewards must be one-dimensional")
    if actions.shape != rewards.shape:
        raise ValueError("actions and rewards must have the same length")
    if not np.issubdtype(actions.dtype, np.integer):
        raise ValueError("actions must be an integer array")
    if actions.dtype == np.bool_ or not (
        np.issubdtype(rewards.dtype, np.integer) or np.issubdtype(rewards.dtype, np.floating)
    ):
        raise ValueError("rewards must be a real numeric array")
    if not np.all(np.isfinite(rewards)):
        raise ValueError("rewards must be finite")

    if not _is_real_number(n_actions) or isinstance(n_actions, float) or n_actions < 1:
        raise ValueError("n_actions must be an integer >= 1")
    n_actions = int(n_actions)

    if step_size is not None:  # noqa: SIM102
        if not _is_real_number(step_size) or not (0 < step_size <= 1):
            raise ValueError("step_size must be None or a number in (0, 1]")

    if not _is_real_number(initial_value) or not np.isfinite(initial_value):
        raise ValueError("initial_value must be a finite real number")

    if actions.size and (actions.min() < 0 or actions.max() >= n_actions):
        raise ValueError("action index out of range")

    q = np.full(n_actions, float(initial_value), dtype=np.float64)
    counts = np.zeros(n_actions, dtype=np.int64)
    for a_raw, r_raw in zip(actions, rewards):  # noqa: B905
        a = int(a_raw)
        r = float(r_raw)
        counts[a] += 1
        if step_size is None:
            step = 1.0 / counts[a]
        else:
            step = float(step_size)
        q[a] = q[a] + step * (r - q[a])
    return q

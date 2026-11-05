from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def action_value_estimates(
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
    raise NotImplementedError

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


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
    raise NotImplementedError

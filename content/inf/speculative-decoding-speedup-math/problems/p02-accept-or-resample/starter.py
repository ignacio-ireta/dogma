from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def accept_or_resample(
    p: NDArray[np.float64],
    q: NDArray[np.float64],
    draft_token: int,
    rng: Any,
) -> int:
    """Return the token one speculative step emits at a single position.

    p: target probabilities, 1-D, length V >= 1, finite, nonnegative, sum 1 within 1e-9.
    q: draft probabilities with the same rules. draft_token: int in [0, V) with q[draft_token] > 0.
    rng: object with random() returning floats in [0, 1). Exactly one draw when accepted, two
    when rejected (accept uniform, then resample uniform). Returns a Python int in [0, V).
    Raise ValueError for invalid arguments, including a zero draft probability at draft_token.
    """
    raise NotImplementedError

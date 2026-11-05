from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def dense_backward(
    X: NDArray[np.float64],
    W: NDArray[np.float64],
    G: NDArray[np.float64],
    reduction: str = "sum",
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return dX, dW, db for Y=X@W+b; mean divides all three by batch size.

    Finite real arrays X:(n,d), W:(d,k), G:(n,k), nonempty dimensions <=64,
    absolute values <=1e6. Cast to float64; reject invalid input with ValueError.
    Never mutate inputs. G is an unreduced upstream gradient.
    """
    raise NotImplementedError

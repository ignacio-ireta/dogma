from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def dropout(
    X: NDArray[np.float64],
    G: NDArray[np.float64],
    p: float,
    training: bool,
    seed: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return output, dX, multiplier for seeded inverted dropout.

    X,G same nonempty rank-2 shape, dimensions <=64, real finite |values|<=1e6.
    0<=p<=0.95, training bool, integer seed 0..2**32-1; otherwise ValueError.
    Train: M=(default_rng(seed).random(X.shape)>=p)/(1-p); output=X*M, dX=G*M.
    Eval: M=ones; output=X.copy(), dX=G.copy(). Float64 outputs, no mutation,
    no global RNG use. Seed regenerates the same mask on each call.
    """
    raise NotImplementedError

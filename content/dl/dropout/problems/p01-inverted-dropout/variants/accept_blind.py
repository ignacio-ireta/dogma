from __future__ import annotations

import math
import numbers

import numpy as np
from numpy.typing import NDArray


def _real_matrix(name: str, value: object) -> NDArray[np.float64]:
    if not isinstance(value, np.ndarray):
        raise ValueError(f"{name} must be a NumPy array")
    if value.dtype == np.bool_ or not (
        np.issubdtype(value.dtype, np.integer) or np.issubdtype(value.dtype, np.floating)
    ):
        raise ValueError(f"{name} must be a real numeric array")
    if value.ndim != 2:
        raise ValueError(f"{name} must be 2-dimensional")
    n, m = value.shape
    if not (1 <= n <= 64 and 1 <= m <= 64):
        raise ValueError(f"{name} dimensions must be between 1 and 64")
    arr = np.array(value, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite entries")
    if np.any(np.abs(arr) > 1e6):
        raise ValueError(f"{name} entries must have absolute value <= 1e6")
    return arr


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
    Xa = _real_matrix("X", X)
    Ga = _real_matrix("G", G)
    if Xa.shape != Ga.shape:
        raise ValueError("X and G must have the same shape")

    if isinstance(p, (bool, np.bool_)) or not isinstance(p, numbers.Real):
        raise ValueError("p must be a real number")
    p_f = float(p)
    if not math.isfinite(p_f) or not (0.0 <= p_f <= 0.95):
        raise ValueError("p must be finite and within [0, 0.95]")

    if not isinstance(training, (bool, np.bool_)):
        raise ValueError("training must be a bool")

    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, numbers.Integral):
        raise ValueError("seed must be an integer")
    seed_i = int(seed)
    if not (0 <= seed_i <= 2**32 - 1):
        raise ValueError("seed must be in [0, 2**32 - 1]")

    if training:
        U = np.random.default_rng(seed_i).random(Xa.shape)
        M = (p_f <= U).astype(np.float64) / (1.0 - p_f)
    else:
        M = np.ones(Xa.shape, dtype=np.float64)

    out = (Xa * M).astype(np.float64)
    dX = (Ga * M).astype(np.float64)
    return out, dX, M.astype(np.float64)

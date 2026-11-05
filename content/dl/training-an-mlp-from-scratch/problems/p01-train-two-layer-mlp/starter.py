from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Params = tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]


def train_mlp(  # noqa: PLR0917 - explicit mathematical training controls
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    hidden: int,
    steps: int,
    lr: float,
    seed: int,
) -> tuple[Params, NDArray[np.float64]]:
    """Train tanh hidden / linear scalar output with full-batch simultaneous SGD.

    X:(n,d), y:(n,1), 1<=n,d<=32, finite real |values|<=10; hidden 1..32,
    steps 0..200, 0<lr<=1, seed integer 0..2**32-1. Reject invalid input with
    ValueError, including a nonfinite loss checkpoint.
    Float64 weights W1:(d,h), W2:(h,1) are successive default_rng(seed)
    standard_normal draws divided by sqrt(fan_in); biases b1:(h,), b2:(1,) zero.
    L=mean((prediction-y)**2)/2. Return ((W1,b1,W2,b2), losses) where losses
    contains initial loss then each post-update loss. Do not mutate inputs or RNG.
    """
    raise NotImplementedError

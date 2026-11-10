from __future__ import annotations

import math
import numbers

import numpy as np
from numpy.typing import NDArray

Params = tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]


def _int_control(name: str, value: object, low: int, high: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be an integer")
    v = int(value)
    if not (low <= v <= high):
        raise ValueError(f"{name} must be between {low} and {high}")
    return v


def _real_array(name: str, value: object, ndim: int) -> NDArray[np.float64]:
    if not isinstance(value, np.ndarray):
        raise ValueError(f"{name} must be a NumPy array")
    if value.dtype == np.bool_ or not (
        np.issubdtype(value.dtype, np.integer) or np.issubdtype(value.dtype, np.floating)
    ):
        raise ValueError(f"{name} must be a real numeric array")
    if value.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional")
    return np.array(value, dtype=np.float64, copy=True)


def train_mlp(  # noqa: PLR0915, PLR0917
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
    Xa = _real_array("X", X, 2)
    ya = _real_array("y", y, 2)
    n, d = Xa.shape
    if not (1 <= n <= 32 and 1 <= d <= 32):
        raise ValueError("X dimensions must be between 1 and 32")
    if ya.shape != (n, 1):
        raise ValueError("y must have shape (n, 1)")
    if not np.all(np.isfinite(Xa)) or np.any(np.abs(Xa) > 10):
        raise ValueError("X entries must be finite with absolute value <= 10")
    if not np.all(np.isfinite(ya)):
        raise ValueError("y entries must be finite")

    h = _int_control("hidden", hidden, 1, 32)
    s = _int_control("steps", steps, 0, 200)
    sd = _int_control("seed", seed, 0, 2**32 - 1)
    if isinstance(lr, (bool, np.bool_)) or not isinstance(lr, numbers.Real):
        raise ValueError("lr must be a real number")
    lr_f = float(lr)
    if not math.isfinite(lr_f) or not (0.0 < lr_f <= 1.0):
        raise ValueError("lr must be finite with 0 < lr <= 1")

    rng = np.random.default_rng(sd)
    W1 = rng.standard_normal((d, h)) / math.sqrt(d)
    W2 = rng.standard_normal((h, 1)) / math.sqrt(h)
    b1 = np.zeros(h, dtype=np.float64)
    b2 = np.zeros(1, dtype=np.float64)

    def loss_and_grads(W1, b1, W2, b2):
        H = np.tanh(Xa @ W1 + b1)
        pred = H @ W2 + b2
        resid = pred - ya
        loss = float(np.mean(resid**2) / 2.0)
        R = resid / n
        gW2 = H.T @ R
        gb2 = R.sum(axis=0)
        dZ = (R @ W2.T) * (1.0 - H * H)
        gW1 = Xa.T @ dZ
        gb1 = dZ.sum(axis=0)
        return loss, (gW1, gb1, gW2, gb2)

    losses = np.empty(s + 1, dtype=np.float64)
    loss, _ = loss_and_grads(W1, b1, W2, b2)
    if not math.isfinite(loss):
        raise ValueError("training reached a nonfinite loss")
    losses[0] = loss

    for step in range(s):
        loss, (gW1, gb1, gW2, gb2) = loss_and_grads(W1, b1, W2, b2)
        W1 = W1 - lr_f * gW1
        b1 = b1 - lr_f * gb1
        W2 = W2 - lr_f * gW2
        b2 = b2 - lr_f * gb2
        new_loss, _ = loss_and_grads(W1, b1, W2, b2)
        if not math.isfinite(new_loss):
            raise ValueError("training reached a nonfinite loss")
        losses[step + 1] = new_loss

    params: Params = (
        np.asarray(W1, dtype=np.float64),
        np.asarray(b1, dtype=np.float64),
        np.asarray(W2, dtype=np.float64),
        np.asarray(b2, dtype=np.float64),
    )
    return params, losses

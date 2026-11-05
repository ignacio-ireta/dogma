from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _matrix(value: NDArray[np.float64], limit: float, size: int) -> NDArray[np.float64]:
    """Validate before casting so complex data cannot silently lose information."""
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError("expected real numeric arrays")
    array = raw.astype(np.float64)
    if array.ndim != 2 or not all(1 <= n <= size for n in array.shape):
        raise ValueError("expected a nonempty bounded matrix")
    if not np.all(np.isfinite(array)) or np.any(np.abs(array) > limit):
        raise ValueError("values outside finite bounds")
    return array


Params = tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]


def _controls(hidden: int, steps: int, lr: float, seed: int) -> None:
    """Reject hyperparameters outside the deliberately small training contract."""
    for value, lo, hi in ((hidden, 1, 32), (steps, 0, 200), (seed, 0, 2**32 - 1)):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or not lo <= value <= hi
        ):
            raise ValueError("integer hyperparameter outside bounds")
    if not isinstance(lr, (int, float)) or not np.isfinite(lr) or not 0 < lr <= 1:
        raise ValueError("learning rate must be in (0,1]")


def train_mlp(  # noqa: PLR0917 - explicit mathematical training controls
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    hidden: int,
    steps: int,
    lr: float,
    seed: int,
) -> tuple[Params, NDArray[np.float64]]:
    """Train tanh/linear regression with simultaneous full-batch SGD.

    X:(n,d), y:(n,1), finite real |values|<=10, dimensions 1..32. hidden 1..32,
    steps 0..200, lr in (0,1], integer seed 0..2**32-1. Invalid input raises
    ValueError, including a nonfinite loss checkpoint.
    Float64 weights are sequential default_rng normal draws scaled
    by inverse sqrt fan_in; biases start at zero. Return (W1,b1,W2,b2) and an
    array of initial then post-update half-mean-squared losses. No input mutation.
    """
    X, y = (_matrix(a, 10.0, 32) for a in (X, y))
    if y.shape != (X.shape[0], 1):
        raise ValueError("target must be a matching column")
    _controls(hidden, steps, lr, seed)
    rng = np.random.default_rng(seed)
    W1 = rng.standard_normal((X.shape[1], hidden)) / np.sqrt(X.shape[1])
    b1 = np.zeros(hidden)
    W2 = rng.standard_normal((hidden, 1)) / np.sqrt(hidden)
    b2 = np.zeros(1)
    losses = np.empty(steps + 1)
    for step in range(steps + 1):
        with np.errstate(over="ignore", invalid="ignore"):
            H = np.tanh(X @ W1 + b1)
            error = H @ W2 + b2 - y
            losses[step] = 0.5 * np.mean(error**2)
        if not np.isfinite(losses[step]):
            raise ValueError("training reached a nonfinite loss checkpoint")
        if step == steps:
            break
        upstream = error / X.shape[0]
        # Cache all gradients before updating either layer's parameters.
        dW2, db2 = H.T @ upstream, upstream.sum(axis=0)
        hidden_grad = (upstream @ W2.T) * (1 - H**2)
        dW1, db1 = X.T @ hidden_grad, hidden_grad.sum(axis=0)
        W1, b1 = W1 - lr * dW1, b1 - lr * db1
        W2, b2 = W2 - lr * dW2, b2 - lr * db2
    return (W1, b1, W2, b2), losses

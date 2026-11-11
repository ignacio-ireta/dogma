from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def huber_loss_and_grad(residuals: ArrayLike, delta: float) -> tuple[float, NDArray[np.float64]]:
    """Wrong on purpose: the gradient is returned as float32. Return (mean Huber loss, gradient of that mean with respect to each residual).

    Per residual r with threshold delta: loss r^2/2 when |r| <= delta, else delta * (|r| - delta/2);
    derivative r when |r| <= delta, else delta * sign(r). The mean loss is (1/n) * sum of the
    losses,
    and gradient_i = derivative_i / n. Raises ValueError for residuals that are not 1-D, empty or
    non-finite, and for delta that is not a finite number greater than 0. Inputs must not be
    mutated.
    """  # noqa: E501
    # np.array copies, so the caller's residuals are never written to.
    r = np.array(residuals, dtype=np.float64)
    if r.ndim != 1:
        raise ValueError(f"residuals must be 1-D, got shape {r.shape}")
    if r.size == 0:
        raise ValueError("need at least one residual")
    if not np.all(np.isfinite(r)):
        raise ValueError("residuals must contain only finite numbers")
    if not (np.isfinite(delta) and delta > 0):
        raise ValueError(f"delta must be a finite number greater than 0, got {delta!r}")

    n = r.size
    # The Huber loss switches from quadratic to linear at |r| = delta. The two pieces agree there:
    # r^2/2 = delta * (delta - delta/2) when |r| = delta, so the loss is continuous.
    inside = np.abs(r) <= delta
    # Mask before evaluating: np.where would square even the huge linear-branch outliers.
    per_residual = np.empty_like(r)
    per_residual[inside] = (0.5 * r[inside]) * r[inside]
    per_residual[~inside] = delta * (np.abs(r[~inside]) - 0.5 * delta)
    # Scale before averaging to avoid an overflowing sum of representable losses.
    scale = float(np.max(per_residual))
    if scale == 0.0:
        loss = 0.0
    elif np.isinf(scale):
        loss = float("inf")
    else:
        loss = float(scale * np.mean(per_residual / scale))

    # The derivative of the loss is psi(r): r inside the threshold, and a capped delta * sign(r)
    # outside it. That cap is why a large residual has a bounded influence on the gradient.
    psi = np.where(inside, r, delta * np.sign(r))
    # The mean loss divides each term by n, so each gradient entry carries 1/n as well.
    grad = (psi / n).astype(np.float32)
    return loss, grad

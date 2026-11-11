from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray


def huber_loss_and_grad(residuals: ArrayLike, delta: float) -> tuple[float, NDArray[np.float64]]:
    try:
        r = np.array(residuals, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("residuals must be numeric") from exc
    if r.ndim != 1 or r.size == 0 or not np.all(np.isfinite(r)):
        raise ValueError("residuals must be a non-empty finite 1-D array")
    try:
        d = float(delta)
    except (TypeError, ValueError) as exc:
        raise ValueError("delta must be a number") from exc
    if not math.isfinite(d) or d <= 0:
        raise ValueError("delta must be a finite number greater than 0")
    n = r.size
    absr = np.abs(r)
    small = absr <= d
    per = np.empty(n, dtype=np.float64)
    per[small] = (0.5 * r[small]) * r[small]
    per[~small] = d * (absr[~small] - d / 2.0)
    loss = float(math.fsum((per / n).tolist()))
    psi = np.where(small, r, d * np.sign(r))
    grad = (psi / n).astype(np.float64)
    return loss, grad

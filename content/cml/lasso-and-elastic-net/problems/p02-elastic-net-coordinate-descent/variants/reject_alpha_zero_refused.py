from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def elastic_net_cd(
    X: ArrayLike,
    y: ArrayLike,
    alpha: float,
    l1_ratio: float,
    *,
    max_iter: int = 10_000,
    tol: float = 1e-8,
) -> NDArray[np.float64]:
    """Wrong on purpose: refuses alpha equal to zero, which is valid.

    Minimises (1/(2n)) ||y - X w||^2 + alpha * l1_ratio * ||w||_1
    + 0.5 * alpha * (1 - l1_ratio) * ||w||^2.
    For coordinate j, with rho_j = (1/n) X_j^T (y - sum_{k != j} X_k w_k) and z_j = (1/n) ||X_j||^2,
    the update is w_j = S(rho_j, alpha * l1_ratio) / (z_j + alpha * (1 - l1_ratio)), where
    S(x, lam) = sign(x) * max(|x| - lam, 0). A zero denominator keeps w_j = 0.

    Raises
    ------
    ValueError
        If X is not 2-D, y is not 1-D with one entry per row, the data is empty or non-finite,
        alpha is negative or non-finite, l1_ratio is outside [0, 1], max_iter < 1, or tol is not
        positive.
    """
    # np.array copies, so the caller's arrays are never written to.
    x_arr = np.array(X, dtype=np.float64)
    y_arr = np.array(y, dtype=np.float64)
    if x_arr.ndim != 2 or x_arr.shape[0] == 0 or x_arr.shape[1] == 0:
        raise ValueError(f"X must be 2-D with at least one row and column, got shape {x_arr.shape}")
    n, d = x_arr.shape
    if y_arr.ndim != 1 or y_arr.shape[0] != n:
        raise ValueError(f"y must be 1-D with {n} entries, got shape {y_arr.shape}")
    if not (np.all(np.isfinite(x_arr)) and np.all(np.isfinite(y_arr))):
        raise ValueError("X and y must contain only finite numbers")
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError(f"alpha must be a finite number >= 0, got {alpha!r}")
    if not (np.isfinite(l1_ratio) and 0.0 <= l1_ratio <= 1.0):
        raise ValueError(f"l1_ratio must lie in [0, 1], got {l1_ratio!r}")
    if max_iter < 1:
        raise ValueError(f"max_iter must be at least 1, got {max_iter!r}")
    if not (np.isfinite(tol) and tol > 0):
        raise ValueError(f"tol must be a finite number greater than 0, got {tol!r}")

    # The L1 part sets the threshold alpha * l1_ratio, and the L2 part adds alpha * (1 - l1_ratio)
    # to the curvature. That extra curvature is what spreads weight across correlated columns.
    threshold = alpha * l1_ratio
    l2_term = alpha * (1.0 - l1_ratio)
    z = np.sum(x_arr**2, axis=0) / n
    w = np.zeros(d)
    residual = y_arr.copy()

    for _ in range(max_iter):
        largest_change = 0.0
        for j in range(d):
            denominator = z[j] + l2_term
            if denominator == 0.0:
                # Zero column with no L2 term: nothing to fit, so the coefficient stays at 0.
                continue
            column = x_arr[:, j]
            residual += column * w[j]
            rho = column @ residual / n
            new_w = np.sign(rho) * max(abs(rho) - threshold, 0.0) / denominator
            residual -= column * new_w
            largest_change = max(largest_change, abs(new_w - w[j]))
            w[j] = new_w
        if largest_change < tol:
            break
    return w

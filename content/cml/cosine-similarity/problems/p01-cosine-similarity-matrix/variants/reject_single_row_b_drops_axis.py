from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def cosine_similarity_matrix(A: ArrayLike, B: ArrayLike) -> NDArray[np.float64]:
    """Wrong on purpose: a single row in B drops the result axis. Return the (n, m) matrix of cosine similarities between rows of ``A`` and ``B``.

    Cosine is the dot product of unit vectors. We normalise every row once, then compare
    all pairs with one matrix product: S = Â B̂ᵀ.
    """  # noqa: E501
    a = np.asarray(A, dtype=np.float64)
    b = np.asarray(B, dtype=np.float64)

    # Validate before any arithmetic. Shape bugs are the most common failure in this code.
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"expected 2-D inputs, got shapes {a.shape} and {b.shape}")
    if a.shape[1] != b.shape[1]:
        raise ValueError(f"feature count mismatch: {a.shape[1]} vs {b.shape[1]}")
    if a.size == 0 or b.size == 0:
        raise ValueError("inputs must have at least one row and one column")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("inputs must contain only finite numbers (no NaN or inf)")

    unit_a, zero_a = _unit_rows(a)
    unit_b, zero_b = _unit_rows(b)

    # Each entry is a dot product of two unit vectors, so it is at most 1 in exact
    # arithmetic. Rounding can push it a hair outside [-1, 1], so clip.
    sims = np.clip(unit_a @ unit_b.T, -1.0, 1.0)

    # Cosine is undefined for a zero vector. We follow scikit-learn and report 0.0.
    sims[zero_a, :] = 0.0
    sims[:, zero_b] = 0.0
    if sims.shape[1] == 1:
        return sims[:, 0]
    return sims


def _unit_rows(x: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Return (rows divided by their L2 norm, boolean mask of zero rows).

    Each row is first divided by its largest absolute entry. Squaring raw entries overflows
    near 1e154 and underflows near 1e-162, so a naive norm is wrong at both ends. After the
    rescale every entry has magnitude at most 1, and the squares are safe.
    """
    peak = np.max(np.abs(x), axis=1, keepdims=True)
    is_zero = peak[:, 0] == 0.0
    # Zero rows keep a divisor of 1 so they stay exactly zero instead of becoming NaN.
    safe_peak = np.where(is_zero[:, None], 1.0, peak)
    scaled = x / safe_peak

    norms = np.sqrt(np.sum(scaled * scaled, axis=1, keepdims=True))
    safe_norms = np.where(is_zero[:, None], 1.0, norms)
    return scaled / safe_norms, is_zero

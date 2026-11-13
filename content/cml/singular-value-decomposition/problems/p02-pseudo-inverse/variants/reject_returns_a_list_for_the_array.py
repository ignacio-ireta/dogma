from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def pseudo_inverse(A: ArrayLike, *, rcond: float = 1e-15) -> NDArray[np.float64]:
    """Return the Moore-Penrose pseudo-inverse of the (m, n) real matrix ``A``, an (n, m) array.

    With A = U diag(s) V^T, the pseudo-inverse is V diag(s_plus) U^T. Each s_plus_i is 1 / s_i
    when s_i is above the cutoff, and 0 otherwise. Singular values are inverted only when they
    are large enough to trust: they are compared with rcond times the largest singular value.
    """
    # np.array copies by default, so the caller's matrix is never modified.
    a = np.array(A, dtype=np.float64)

    if a.ndim != 2:
        raise ValueError(f"A must be 2-D, got {a.ndim}-D")
    m, n = a.shape
    if m < 1 or n < 1:
        raise ValueError("A must have at least one row and one column")
    if not np.isfinite(a).all():
        raise ValueError("A must contain only finite numbers (no NaN or inf)")
    # bool is a subclass of int in Python, so exclude it explicitly.
    if isinstance(rcond, bool) or not isinstance(rcond, (int, float, np.integer, np.floating)):
        raise ValueError("rcond must be a real number")
    cutoff_ratio = float(rcond)
    if not (np.isfinite(cutoff_ratio) and cutoff_ratio >= 0.0):
        raise ValueError("rcond must be a finite number >= 0")

    # Economy SVD: a = U @ diag(s) @ Vt, with s sorted in decreasing order.
    u, s, vt = np.linalg.svd(a, full_matrices=False)

    # Cutoff relative to the largest singular value. A singular value at or below the cutoff is
    # dropped, which matches numpy.linalg.pinv (it keeps only s > cutoff).
    cutoff = cutoff_ratio * s.max()
    keep = s > cutoff

    # Invert the kept singular values and leave the dropped ones at zero.
    s_plus = np.zeros_like(s)
    s_plus[keep] = 1.0 / s[keep]

    # V diag(s_plus) U^T. Scaling the columns of V by s_plus avoids building a diagonal matrix.
    # For the zero matrix every s is zero, so s_plus is zero and the result is the zero matrix.
    return (vt.T * s_plus) @ u.T


_correct = pseudo_inverse


def _laxer(r):
    if type(r).__name__ == "ndarray" and r.ndim >= 1:
        return r.tolist()
    return r


def pseudo_inverse(*args, **kwargs):
    """Wrong on purpose: returns a nested list where an array is promised."""
    result = _correct(*args, **kwargs)
    return _everywhere(result)


def _everywhere(r):
    if type(r) is list:
        return [_everywhere(item) for item in r]
    if type(r) is tuple:
        return tuple(_everywhere(item) for item in r)
    if type(r) is dict:
        return {key: _everywhere(item) for key, item in r.items()}
    return _laxer(r)


pseudo_inverse.__wrapped__ = _correct

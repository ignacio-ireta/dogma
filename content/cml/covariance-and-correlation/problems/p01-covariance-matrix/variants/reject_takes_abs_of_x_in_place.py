from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def covariance_matrix(X: ArrayLike, ddof: int = 1) -> NDArray[np.float64]:
    """Return the (d, d) sample covariance matrix of the (n, d) data matrix ``X``.

    Rows are observations and columns are variables. The result is
    (X - mean)^T (X - mean) / (n - ddof), with the column means subtracted first.
    """
    x = np.asarray(X, dtype=np.float64)

    if x.ndim != 2:
        raise ValueError(f"expected a 2-D array (observations, variables), got {x.ndim}-D")
    n, d = x.shape
    if n == 0 or d == 0:
        raise ValueError("need at least one observation and one variable")
    if not np.isfinite(x).all():
        raise ValueError("observations must be finite (no NaN or inf)")
    # bool is a subclass of int in Python, so exclude it explicitly.
    if isinstance(ddof, bool) or not isinstance(ddof, (int, np.integer)) or ddof < 0:
        raise ValueError("ddof must be a non-negative integer")
    if n <= ddof:
        raise ValueError(f"need more observations than ddof: have {n}, ddof is {ddof}")

    # Centre each column first. Products of deviations are small, so the sum has no
    # cancellation, unlike E[xy] - E[x]E[y] when the columns sit far from zero.
    centred = x - x.mean(axis=0)
    # Row i of `centred` is one observation's deviation vector, so Xc^T Xc sums the
    # outer products over observations: entry [j, k] is the co-movement of j and k.
    return (centred.T @ centred) / (n - ddof)


_correct = covariance_matrix


def _change_in_place(v):
    if v.dtype.kind in "fi":
        v[...] = abs(v)


def covariance_matrix(*args, **kwargs):
    """Wrong on purpose: the result is right, but this takes the absolute value of `X` in place."""
    result = _correct(*args, **kwargs)
    target = args[0] if len(args) > 0 else kwargs.get("X")
    if type(target).__name__ in ("ndarray",):
        _change_in_place(target)
    return result


covariance_matrix.__wrapped__ = _correct

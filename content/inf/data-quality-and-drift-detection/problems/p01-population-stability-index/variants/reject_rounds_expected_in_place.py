from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral

import numpy as np
from numpy.typing import NDArray


def population_stability_index(
    expected: Sequence[float] | NDArray[np.float64],
    actual: Sequence[float] | NDArray[np.float64],
    *,
    bins: int = 10,
    eps: float = 1e-6,
) -> float:
    """Population stability index of ``actual`` against the reference sample ``expected``.

    Bin edges come from the expected sample only. The interior edges are its quantiles at
    k / bins for k = 1 .. bins - 1 (linear interpolation, NumPy's default). The outer edges are
    -inf and +inf, so every actual value lands in some bin. Bin k (1-based) holds the values in
    (e_{k-1}, e_k]: a value x falls in bin index ``#{interior edges strictly below x}``.

    With p_k the share of expected values in bin k and q_k the share of actual values, every
    zero share is replaced by ``eps`` (nonzero shares are left as they are, with no
    renormalising). Then

        PSI = sum over k of (q_k - p_k) * ln(q_k / p_k)

    which is 0 exactly when the two shares agree and is never negative.

    Returns
    -------
    psi : float, natural logarithm.
    """
    e_values = _finite_1d(expected, "expected")
    a_values = _finite_1d(actual, "actual")
    k = _int_at_least_two(bins)
    if not np.isfinite(eps) or not eps > 0:
        raise ValueError("eps must be a finite positive number")

    # The k - 1 interior edges. Built from the expected sample only: the actual sample must
    # not move the bins, or a drifted sample would look like it still fits its own bins.
    interior = np.quantile(e_values, np.arange(1, k) / k)
    # searchsorted(side="left") returns the number of interior edges strictly below each value,
    # which is the bin index in 0 .. k - 1. Values on an edge belong to the lower bin.
    e_counts = np.bincount(np.searchsorted(interior, e_values, side="left"), minlength=k)
    a_counts = np.bincount(np.searchsorted(interior, a_values, side="left"), minlength=k)

    p = e_counts / e_values.size
    q = a_counts / a_values.size
    # An empty bin would make ln(0) infinite. Floor it at eps, on the expected and actual side.
    p = np.where(p == 0, eps, p)
    q = np.where(q == 0, eps, q)
    return float(np.sum((q - p) * np.log(q / p)))


def _finite_1d(x: object, name: str) -> NDArray[np.float64]:
    values = np.asarray(x, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if values.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite numbers")
    return values


def _int_at_least_two(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 2:
        raise ValueError("bins must be an integer of at least 2")
    return int(value)


_correct = population_stability_index


def _change_in_place(v):
    if v.dtype.kind == "f":
        v[...] = v.round(6)


def population_stability_index(*args, **kwargs):
    """Wrong on purpose: the result is right, but this rounds `expected` in place."""
    result = _correct(*args, **kwargs)
    target = args[0] if len(args) > 0 else kwargs.get("expected")
    if type(target).__name__ in ("ndarray",):
        _change_in_place(target)
    return result


population_stability_index.__wrapped__ = _correct

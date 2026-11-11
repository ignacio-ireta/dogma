from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats


def welch_t_test(a: ArrayLike, b: ArrayLike) -> tuple[float, float, float]:
    """Return ``(t, df, p_two_sided)`` for Welch's two-sample t-test of ``a`` against ``b``.

    Parameters
    ----------
    a, b : 1-D array-likes of n >= 2 finite real numbers each (ints allowed).

    Returns
    -------
    (t, df, p) where ``t = (mean(a) - mean(b)) / sqrt(va + vb)``, ``va = var(a, ddof=1) / n_a``,
    ``vb = var(b, ddof=1) / n_b``, ``df = (va + vb)**2 / (va**2 / (n_a - 1) + vb**2 / (n_b - 1))``
    (Welch-Satterthwaite), and ``p = 2 * P(T_df >= |t|)``.

    Raises
    ------
    ValueError
        If an input is not 1-D, has fewer than two values, contains NaN or infinity, or if both
        samples are constant (the standard error is zero).
    """
    a_vals = np.asarray(a, dtype=np.float64)
    b_vals = np.asarray(b, dtype=np.float64)
    for name, vals in (("a", a_vals), ("b", b_vals)):
        if vals.ndim != 1:
            raise ValueError(f"sample {name} must be 1-D, got shape {vals.shape}")
        if vals.size < 2:
            raise ValueError(f"sample {name} needs at least two observations")
        if not np.all(np.isfinite(vals)):
            raise ValueError("observations must be finite (no NaN or infinity)")
    a_constant = bool(np.all(a_vals == a_vals[0]))
    b_constant = bool(np.all(b_vals == b_vals[0]))
    if a_constant and b_constant:
        raise ValueError("both samples are constant: the t statistic is undefined")
    pooled_mean = float(np.concatenate([a_vals, b_vals]).mean())
    centred_a = a_vals - pooled_mean
    centred_b = b_vals - pooled_mean
    scale = max(float(np.min(np.abs(centred_a))), float(np.max(np.abs(centred_b))))
    if scale == 0.0:
        raise ValueError("both samples are constant: the t statistic is undefined")
    u = centred_a / scale
    v = centred_b / scale
    n_a, n_b = (u.size, v.size)
    var_a = 0.0 if a_constant else float(u.var(ddof=1)) / n_a
    var_b = 0.0 if b_constant else float(v.var(ddof=1)) / n_b
    se_squared = var_a + var_b
    if se_squared == 0.0:
        raise ValueError("both samples are constant: the t statistic is undefined")
    t_stat = (float(u.mean()) - float(v.mean())) / np.sqrt(se_squared)
    df = se_squared**2 / (var_a**2 / (n_a - 1) + var_b**2 / (n_b - 1))
    p_value = 2.0 * float(stats.t.sf(abs(t_stat), df))
    return (float(t_stat), float(df), p_value)

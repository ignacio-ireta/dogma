import numpy as np
from scipy import stats


def welch_t_test(a, b):
    xs = np.asarray(a, dtype=np.float64)
    ys = np.asarray(b, dtype=np.float64)
    for vals in (xs, ys):
        if vals.ndim != 1:
            raise ValueError("each sample must be one-dimensional")
        if vals.size < 2:
            raise ValueError("each sample needs at least two observations")
        if not np.all(np.isfinite(vals)):
            raise ValueError("observations must be finite")
    if np.all(xs == xs[0]) and np.all(ys == ys[0]):
        raise ValueError("both samples are constant: the t statistic is undefined")
    # centre on the pooled mean and divide by the largest deviation: shift- and scale-invariant
    centre = np.concatenate([xs, ys]).mean()
    xc = xs - centre
    yc = ys - centre
    unit = max(float(np.max(np.abs(xc))), float(np.max(np.abs(yc))))
    u = xc / unit
    v = yc / unit
    var_a = u.var(ddof=1) / u.size
    var_b = v.var(ddof=1) / v.size
    se2 = var_a + var_b
    t = (u.mean() - v.mean()) / np.sqrt(se2)
    # Welch-Satterthwaite degrees of freedom, with each group's own variance term
    df = se2**2 / (var_a**2 / (u.size - 1) + var_b**2 / (v.size - 1))
    p = 2.0 * stats.t.sf(abs(t), df)
    return float(t), float(df), float(p)

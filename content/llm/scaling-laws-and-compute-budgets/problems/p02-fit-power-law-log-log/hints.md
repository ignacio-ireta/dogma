## Hint 1 — concept

A power law is a straight line on log-log axes. Fitting it means fitting a line to the logs of the
data, with the usual least-squares rules. The two constants of the power law come back from the
line's intercept and slope.

## Hint 2 — formula / approach

Take `u = log x` and `v = log y`. The slope is the covariance of `u` and `v` divided by the variance
of `u`, both computed from the centred values. The intercept is the mean of `v` minus the slope
times the mean of `u`. The coefficient `a` is `exp(intercept)`. Check the inputs first: same length,
at least two points, one dimension, finite values, strictly positive values, and `x` not constant.

## Hint 3 — code shape

Convert both inputs with `np.asarray(..., dtype=np.float64)`. Run the validation checks. Compute
`u = np.log(x)` and `v = np.log(y)`, then `du = u - u.mean()` and `dv = v - v.mean()`. The slope is
`np.sum(du * dv) / np.sum(du * du)`. Return `(math.exp(c), b)` where `c = v.mean() - b * u.mean()`.

# Mean confidence interval

Implement `mean_confidence_interval(x, confidence=0.95)`. It returns the sample mean and the endpoints
of its two-sided Student-t confidence interval as `(mean, lo, hi)`. The interval is
`mean ± t · s / √n`, where `s` is the sample standard deviation with `ddof = 1`, and `t` is the
`(1 + confidence) / 2` quantile of Student's t distribution with `n - 1` degrees of freedom.

## Contract

- `x`: 1-D sequence of real numbers (a list or a NumPy array) of length `n ≥ 2`. Integers are allowed.
  Every value must be finite.
- `confidence`: a float strictly between 0 and 1. Default `0.95`.
- Returns: a tuple `(mean, lo, hi)` of three floats, with `lo ≤ mean ≤ hi`.
- Raises `ValueError` when `x` is not 1-D, when it has fewer than two values, when a value is NaN or
  infinite, or when `confidence` is not strictly between 0 and 1.
  A constant sample is valid and gives `lo == hi == mean`.
- Must not mutate its input.

## Examples

```python
>>> mean_confidence_interval([1, 2, 3, 4, 5])
(3.0, 1.0367568385, 4.9632431615)     # s = 1.5811, SE = 0.7071, t(0.975, df=4) = 2.7764
>>> mean_confidence_interval([2, 2, 2])
(2.0, 2.0, 2.0)
>>> mean_confidence_interval([7.0])
ValueError: need at least two observations to estimate a standard error
```

## Constraints

- Allowed: NumPy, and SciPy's Student-t distribution (for example `scipy.stats.t.ppf`).
- Not allowed: `scipy.stats.t.interval` (it returns the interval directly, which is the task) and
  `statsmodels`.

## Done looks like

All public tests pass, then **Submit** passes the hidden tests. Those compare the endpoints with a
Student-t reference on seeded samples of several sizes and confidence levels. They check the t quantile's
degrees of freedom and the `ddof = 1` standard error on hand-checkable samples, including `n = 2`. They
check that a higher confidence level gives a wider interval and that the interval covers the true mean at
the nominal rate in a seeded simulation with `n = 5`. They also cover a large common offset, every kind
of invalid input, and inputs that must be left unchanged.

The hidden tests draw their bulk inputs from the real dataset `california_housing` (see
`docs/DATASETS.md`).

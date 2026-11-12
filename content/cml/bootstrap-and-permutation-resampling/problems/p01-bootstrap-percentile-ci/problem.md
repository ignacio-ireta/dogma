# Bootstrap percentile interval

Implement `bootstrap_percentile_ci(x, statistic, *, n_resamples=2000, confidence=0.95, seed)`. It
returns the percentile bootstrap interval `(lo, hi)` for `statistic` computed on the sample `x`. The
interval is the pair of quantiles of the bootstrap replicates, one replicate per resample.

## Contract

- `x`: 1-D sequence of finite real numbers (a list or a NumPy array) of length `n ≥ 1`. Integers are
  allowed, and so are float32 arrays: `statistic` always receives float64 resamples.
- `statistic`: a function that takes a 1-D float array and returns a float. It is called once per
  resample, on a resample of length `n`.
- `n_resamples`: integer `≥ 1`. Default `2000`.
- `confidence`: a float strictly between 0 and 1. Default `0.95`.
- `seed`: required integer. The result depends only on `x`, `statistic`, `n_resamples`, `confidence`
  and `seed`. Do not use unseeded or global randomness.
- Returns: a tuple `(lo, hi)` of two floats, with `lo ≤ hi`.
- **Resampling protocol** (the grader checks against this exact draw):
  1. `rng = np.random.default_rng(seed)`
  2. `idx = rng.integers(0, n, size=(n_resamples, n))`. Row `b` is resample `b`, drawn with replacement.
     Repeated values are separate rows: a value that appears `k` times in `x` has `k` rows to be drawn from.
  3. Replicate `b` is `statistic(x[idx[b]])`.
  4. With `alpha = 1 - confidence`: `lo, hi = np.quantile(replicates, [alpha / 2, 1 - alpha / 2], method="linear")`.
- Raises `ValueError` when `x` is not 1-D, when it is empty, when it contains NaN or infinity, when
  `n_resamples < 1`, or when `confidence` is not strictly between 0 and 1.
- Must not mutate its input.

## Examples

```python
>>> bootstrap_percentile_ci([1, 2, 3, 4, 5], np.mean, seed=0)
(1.8, 4.2)
>>> bootstrap_percentile_ci([7, 7, 7], np.mean, seed=0)     # no spread, so no width
(7.0, 7.0)
>>> bootstrap_percentile_ci([1, 2, 3], np.mean, n_resamples=0, seed=0)
ValueError: n_resamples must be at least 1
```

On real data, the `iris` sepal lengths (150 values rounded to 0.1 cm, only 35 distinct) keep their repeats, so the
resampled means are drawn from the full sample, not from its distinct values.

## Constraints

- Allowed: NumPy (`np.random.default_rng`, `rng.integers`, `np.quantile`, indexing, …).
- Not allowed: `scipy.stats.bootstrap`, which computes an interval for you. The task is to write the
  resampling loop and the quantiles yourself.

## Done looks like

All public tests pass, then **Submit** passes the hidden tests. Those compare the interval with a
reference that follows the protocol above exactly, for several samples, seeds, confidence levels, and
the statistics mean, median and maximum. They check that the interval is nested as confidence grows, that
a constant sample gives a point interval, that the same seed reproduces the result, that the statistic is called once per resample on a one-dimensional
resample of length `n`, that the interval on repeated real measurements (iris sepal lengths, also stored as float32) follows the protocol and does not resample distinct values, that invalid
arguments raise `ValueError` and that the input is left unchanged.

The hidden tests draw their bulk inputs from the real datasets `california_housing` and `iris` (see
`docs/DATASETS.md`).

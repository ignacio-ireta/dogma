## Hint 1 — concept

The bootstrap treats the sample as a stand-in for the population. Resampling it with replacement
shows how much the statistic would move between samples of the same size. The interval is read
straight off the spread of those recomputed statistics.

## Hint 2 — formula / approach

Draw an index matrix of shape `(n_resamples, n)` with replacement, using a generator seeded with
`seed`. Compute `statistic` on each row's resample. Take the `alpha / 2` and `1 - alpha / 2` quantiles
of the replicates, with `alpha = 1 - confidence`, using `method="linear"`.

## Hint 3 — code shape

`rng = np.random.default_rng(seed)`, `idx = rng.integers(0, n, size=(n_resamples, n))`, then
`reps = np.array([statistic(values[row]) for row in idx])`. Finish with
`lo, hi = np.quantile(reps, [alpha / 2, 1 - alpha / 2], method="linear")` and return them as floats.

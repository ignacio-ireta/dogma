# Two-sample Kolmogorov-Smirnov drift check

Implement `ks_drift(reference, current, *, alpha=0.05)`. It tests whether `current` could have been drawn from the
same distribution as `reference`, using the two-sample Kolmogorov-Smirnov statistic. It returns the statistic D, an
approximate p-value, and a flag that is True when the p-value is below `alpha`. The statistic needs no assumption
about the shape of the distribution, which is why drift monitors use it on continuous features.

## Contract

- `reference`, `current`: 1-D sequences of finite real numbers, both non-empty. Integers are allowed. Must not be
  mutated.
- `alpha`: float strictly between 0 and 1.
- Statistic: D = max over x of |F_ref(x) - F_cur(x)|, where F is the empirical CDF of each sample. Evaluate it at
  every pooled value, counting all tied values together. D is two-sided, so a shift in either direction counts.
- P-value: with n = len(reference), m = len(current), and effective size n_e = n * m / (n + m),

      lambda = sqrt(n_e) * D
      p = 2 * sum over k >= 1 of (-1)^(k-1) * exp(-2 k^2 lambda^2),   with p = 1 when lambda = 0

  This is the limiting Kolmogorov distribution. It is an approximation: it is accurate for samples of about a hundred
  values or more, and it is not exact for very small samples. `scipy.stats.ks_2samp` does not return this p-value:
  its default computes an exact finite-sample p-value when both samples have fewer than 10000 values, and
  its `method='asymp'` uses the Kolmogorov distribution at `round(n_e)` rather than at `sqrt(n_e) * D`.
- `drifted`: True when p < alpha, otherwise False.
- Returns: `(statistic, p_value, drifted)`, two floats and a bool.
- Raises `ValueError` when an input is not 1-D, is empty, or has NaN or infinity, and when `alpha` is not strictly
  between 0 and 1.

## Examples

```python
>>> ks_drift([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
(0.5, 0.1640791977266521, False)
```

```python
>>> ks_drift([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4])
(0.5, 0.1640791977266521, False)
```

The second example shifts the current sample to the left. A check that only looks for a current sample that is
larger would miss it.

## Constraints

- Allowed: NumPy, the Python standard library, and `scipy.stats.kstwobign` (the limiting Kolmogorov distribution) if
  you prefer it to your own series for the p-value.
- Not allowed: `scipy.stats.ks_2samp` and `scipy.stats.kstest`. Each runs the test for you, and neither returns the
  p-value defined above.

## Done looks like

All public tests pass, then **Submit** passes the hidden tests. Those compare the statistic with
`scipy.stats.ks_2samp` on seeded samples of several sizes, including heavy ties and unequal sizes. They check the
p-value against the limiting Kolmogorov distribution at the effective size, with unequal sample sizes, and they
distinguish the effective size n * m / (n + m) from n alone. They check that identical samples give D = 0, a p-value of
1 and no drift, that a shift is detected in either direction, that disjoint samples give D = 1, that tied values are
counted as blocks, and that the drift flag follows p < alpha. The tests also cover an invalid alpha, an empty or
non-finite input, a two-dimensional input, integer input, and inputs that must not be mutated.

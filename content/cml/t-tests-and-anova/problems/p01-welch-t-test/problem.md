# Welch's t-test

Implement `welch_t_test(a, b)`. It runs the two-sample t-test that does not assume equal variances and
returns `(t, df, p_two_sided)`. The degrees of freedom come from the Welch–Satterthwaite approximation, and
the p-value is two-sided.

## Contract

- `a`, `b`: 1-D sequences of finite real numbers (lists or NumPy arrays), each of length `n ≥ 2`.
  Integers are allowed.
- Let `va = var(a, ddof=1) / n_a` and `vb = var(b, ddof=1) / n_b`. Then
  - `t = (mean(a) − mean(b)) / sqrt(va + vb)`
  - `df = (va + vb)² / (va² / (n_a − 1) + vb² / (n_b − 1))`
  - `p_two_sided = 2 · P(T_df ≥ |t|)`, where `T_df` is Student's t with `df` degrees of freedom.
- Returns: a tuple `(t, df, p)` of three floats.
- Raises `ValueError` when an input is not 1-D, when a group has fewer than two values, when a value is NaN
  or infinite, or when both samples are constant (the standard error is zero, so `t` is undefined).
  One constant sample with a varying other sample is valid.
- Must not mutate its inputs.

## Examples

```python
>>> welch_t_test([1, 2, 3, 4, 5], [2, 4, 6, 8, 10, 12, 14])
(-2.8097574347, 8.0371057514, 0.0227472553)
>>> welch_t_test([1, 2, 3], [1, 2, 3])        # equal means, equal sizes and spreads
(0.0, 4.0, 1.0)
>>> welch_t_test([1, 1, 1], [2, 2, 2, 2])     # both samples constant
ValueError: both samples are constant: the t statistic is undefined
```

## Constraints

- Allowed: NumPy, and SciPy's Student-t distribution (for example `scipy.stats.t.sf`).
- Not allowed: `scipy.stats.ttest_ind` (it runs the whole test for you) and `statsmodels`.

## Done looks like

All public tests pass, then **Submit** passes the hidden tests. Those compare `t`, `df` and `p` with
`scipy.stats.ttest_ind(equal_var=False)` on seeded samples with unequal sizes and spreads. They check that
unequal variances use Welch's degrees of freedom and not the pooled `n_a + n_b − 2`, that equal sizes and
spreads give `df = 2(n − 1)`, that the test is two-sided and invariant to swapping the groups, shifting and
rescaling (including data at very large or very small magnitudes), that a very small two-sided p-value is computed in the tail and does not round to zero, that one constant sample is allowed,
that invalid inputs raise `ValueError` (with the constant-samples case named), and that inputs are left
unchanged.

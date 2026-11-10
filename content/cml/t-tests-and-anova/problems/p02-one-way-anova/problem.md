# One-way ANOVA

Implement `one_way_anova(*groups)`. It runs the one-way analysis of variance across `k ≥ 2` groups and returns
`(F, p)`. The statistic compares the variation between group means with the variation within groups, and the
p-value is the upper tail of the F distribution.

## Contract

- `*groups`: at least two 1-D sequences of finite real numbers (lists or NumPy arrays). Each group is
  non-empty. A group of one value is allowed. Integers are allowed, and so are float32 arrays: every statistic is computed in float64 on the stored values.
- Let `N` be the total number of observations and `n_i` the size of group `i`, with group mean `m_i` and
  grand mean `m` (mean of all `N` values):
  - `SSB = Σ n_i (m_i − m)²`, and `SSW = Σ_i Σ_j (x_ij − m_i)²`
  - `df1 = k − 1`, `df2 = N − k`
  - `F = (SSB / df1) / (SSW / df2)`
  - `p = P(F_{df1, df2} ≥ F)`, the upper tail of the F distribution
- Returns: a tuple `(F, p)` of two floats.
- Raises `ValueError` when fewer than two groups are given, when a group is empty or not 1-D, when a value is
  NaN or infinite, when `N ≤ k` (there are no within-group degrees of freedom), or when `SSW = 0` (the
  within-group variation is zero and `F` is undefined).
- Must not mutate its inputs.

## Examples

```python
>>> one_way_anova([1, 2, 3], [4, 5, 6])
(13.5, 0.0213116411)
>>> one_way_anova([4.1, 5.0, 4.6, 4.8], [5.9, 6.2, 5.5, 6.4], [5.0, 4.9, 5.3, 5.1])
(17.7814070352, 0.0007476779)
>>> one_way_anova([1, 1], [2, 2])
ValueError: within-group variation is zero: F is undefined
>>> one_way_anova([1, 2, 3])
ValueError: need at least two groups
```

On real data, the three species of `iris` (50 sepal lengths each, stored as float32) must give scipy's F and p on the
same stored values, with spreads that grow from setosa to virginica.

## Constraints

- Allowed: NumPy, and SciPy's F distribution (for example `scipy.stats.f.sf`).
- Not allowed: `scipy.stats.f_oneway` (it runs the whole test for you) and `statsmodels`.

## Done looks like

All public tests pass, then **Submit** passes the hidden tests. Those compare `F` and `p` with
`scipy.stats.f_oneway` on seeded groups with unequal sizes and several group counts. They check that with two
groups `F` equals the square of the pooled-variance t statistic, that a very small p-value is computed in the
upper tail and not as one minus the CDF, that float32 storage of real measurements (the iris species) gives scipy's result, that a singleton group is allowed, that shifting, rescaling and
reordering the groups changes nothing, that a large common offset is handled, that invalid inputs raise
`ValueError` (including the degrees-of-freedom and zero-variation cases), and that inputs are left unchanged.

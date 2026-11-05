# Components for a variance threshold

Implement `components_for_variance(X, threshold)`, which returns how many principal components are
needed to reach a given fraction of the total variance. It returns a count, not the components. Use the
same definitions as PCA: the columns are centred, sample variances have divisor `n - 1`, and each
ratio divides by the total variance of all columns.

## Contract

- `X`: 2-D array-like of real numbers, shape `(n, d)` with `n >= 2` and `d >= 1`. Integers allowed.
  Every entry is finite.
- `threshold`: a real number in `(0, 1]`. Booleans are rejected.
- Let `r_i` be the explained-variance ratio of the `i`-th principal component, with components in
  decreasing order of variance. Returns the smallest `k >= 1` such that `r_1 + ... + r_k >= threshold`,
  as an `int`.
- Ties and rounding: the comparison allows a margin of `1e-12`. A cumulative ratio that equals
  `threshold` in exact arithmetic therefore counts as reaching it, even when rounding leaves it a few
  units below. scikit-learn's `PCA(n_components=threshold)` uses a strict `>`, so on such a tie it returns
  one more component. This function does not follow that rule.
- Raises `ValueError` when `X` is not 2-D, has fewer than two rows or no columns, contains NaN or
  infinity, when `threshold` is not a real number in `(0, 1]`, or when every column is constant. A
  constant matrix has zero total variance, so no ratio exists.
- Must not mutate its input.

## Examples

```python
>>> X = [[-3, -1], [0, -2], [0, 3]]        # the two ratios are 0.75 and 0.25
>>> components_for_variance(X, 0.7)
1
>>> components_for_variance(X, 0.8)
2
>>> components_for_variance(X, 0.75)       # exact tie: 0.75 reaches 0.75
1
>>> components_for_variance(X, 0.0)        # threshold must lie in (0, 1]
ValueError
```

## Constraints

- Allowed: NumPy array operations, including `numpy.linalg.svd` and `numpy.linalg.eigh`.
- Not allowed: `sklearn` and `scipy`.

## Done looks like

All public tests pass, then **Submit** passes the hidden tests. Those check agreement with scikit-learn
on random data at thresholds away from ties; minimality and reach, checked against an independent
eigenvalue computation; monotonicity in the threshold; a threshold of one on rank-deficient data; exact
ties, both a symmetric case and the worked example; a large common offset that an uncentred computation
gets wrong; constant data; threshold, shape and boolean errors; integer input; and inputs left unchanged.

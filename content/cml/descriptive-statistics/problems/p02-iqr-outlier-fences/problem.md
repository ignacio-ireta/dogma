# IQR outlier fences

Implement `iqr_fences(xs, k=1.5)`. It returns Tukey's fences, the two thresholds outside of which a
value counts as an outlier. The fences are `Q1 − k·IQR` and `Q3 + k·IQR`, where `Q1` and `Q3` are
the first and third quartiles and `IQR = Q3 − Q1`. Quartiles use linear interpolation between
order statistics: for sorted values `s` and a quantile `q`, the position is `h = q·(n − 1)`, and the
quantile is `s[⌊h⌋] + (h − ⌊h⌋)·(s[⌊h⌋+1] − s[⌊h⌋])`, with the last index capped at `n − 1`.

## Contract

- `xs`: 1-D array-like of finite real numbers, length n ≥ 1. Integers allowed. Duplicates allowed.
- `k`: non-negative number, default `1.5`.
- Returns: `(lower, upper)`, two Python floats, with `lower = Q1 − k·IQR` and `upper = Q3 + k·IQR`.
- Raises `ValueError` when `xs` is not 1-D, when it is empty, when it contains NaN or infinity, or when
  `k` is negative.
- Must not mutate `xs`, and must not sort it in place.

## Examples

```python
>>> iqr_fences([1, 3, 4, 8, 9])
(-4.5, 15.5)            # Q1 = 3, Q3 = 8, IQR = 5
>>> iqr_fences([1, 3, 4, 8, 9, 100])
(-5.0, 17.0)            # Q1 = 3.25, Q3 = 8.75: the outlier 100 lies above 17
>>> iqr_fences([1, 3, 4, 8, 9], k=0)
(3.0, 8.0)              # with k = 0 the fences are the quartiles themselves
>>> iqr_fences([])
ValueError              # no quartiles of zero values
```

## Constraints

- Allowed: NumPy arrays and the usual operations (`np.asarray`, `np.sort`, `np.isfinite`, …), and
  `math.floor`.
- Not allowed: `numpy.quantile`, `numpy.percentile`, `statistics` and `scipy`. Write the interpolation
  yourself, because the point is the formula.

## Done looks like

All public tests pass, then **Submit** passes the hidden tests. Those check agreement with NumPy's linear
quantiles on random data of many sizes and several values of `k`; a property test on random lists; the
robustness property (replacing the largest value by an extreme one leaves the fences unchanged, while
the mean moves a lot); a single value, constant data and duplicates; `k` = 0 and a wider `k`; monotonicity
in `k`; shift and scale equivariance; invalid input (empty, 2-D, NaN or infinity, negative `k`); integer
input giving floats; and the caller's array left unchanged and unsorted.

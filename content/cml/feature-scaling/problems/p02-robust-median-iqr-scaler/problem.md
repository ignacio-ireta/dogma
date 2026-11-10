# Robust median and IQR scaler

Implement `robust_standardize(train, values)`. Learn one median and one interquartile range per
training feature, then transform `values` with them. Because the median and the IQR ignore the
extreme tails, a few outliers in the training rows barely move the fitted coordinates. The
quartiles are your own code, so the interpolation rule is part of the contract.

## Contract

- `train`: real numeric array-like of shape `(n, d)`, with `n >= 1` and `d >= 1`.
- `values`: real numeric array-like of shape `(m, d)`, with `m >= 0`.
- Entries must be finite. Integers are allowed.
- Returns a tuple `(transformed, center, scale)` of float64 NumPy arrays with shapes `(m, d)`,
  `(d,)` and `(d,)`.
- `center` is the training median of each column, with the usual median of an even count (the
  mean of the two middle values).
- Quartiles use linear interpolation. For a column sorted ascending as `s` with `n` values, and a
  quantile `q`, let `h = q * (n - 1)`, `lo = floor(h)`, `hi = min(lo + 1, n - 1)`. Then the quantile
  is `s[lo] + (h - lo) * (s[hi] - s[lo])` in exact arithmetic. Evaluate it so that it stays finite
  whenever that value is finite: the textbook form overflows for opposite extremes such as `-1e308`
  and `1e308`, whose median is 0. The form `(1 - (h - lo)) * s[lo] + (h - lo) * s[hi]` does not. The quartiles are `q = 0.25` and `q = 0.75`.
- `scale` is `Q3 - Q1` for each column. A column whose scale is below ten times the float64 machine
  epsilon gets scale `1.0`. This is the rule scikit-learn's `RobustScaler` uses. It applies to a
  column that still varies but whose middle half is constant (at least about half of its values
  equal), so its IQR is zero: it gets scale `1.0` too, not its standard deviation.
- Transform with `(values - center) / scale`. Never fit anything on `values`.
- Raise `ValueError` for wrong ranks, empty training dimensions, mismatched feature counts, and
  nonfinite entries.
- Must not mutate either input.

## Examples

```python
robust_standardize([[1, 10], [3, 14], [5, 18], [9, 22]], [[5, 8]])
# (array([[ 0.28571429, -1.33333333]]), array([ 4., 16.]), array([3.5, 6. ]))
```

```python
robust_standardize([[4, 1], [4, 1], [4, 1]], [[7, 2]])
# (array([[3., 1.]]), array([4., 1.]), array([1., 1.]))
```

On raw `digits` (1797 × 64, pixels 0 to 16), 20 pixel columns have IQR zero. Three are constant and
17 still vary. All 20 get scale `1.0`; the other 44 columns get their IQR, the same as
scikit-learn's `RobustScaler`.

## Constraints

Allowed: Python and NumPy array operations, including `np.sort`, `np.median` and `np.floor`.
Not allowed: `sklearn`, `scipy`, `statistics`, and the NumPy quantile functions
`numpy.percentile`, `numpy.quantile`, `numpy.nanpercentile` and `numpy.nanquantile`. Write the
interpolation yourself; it is the point of the exercise.

## Done looks like

All public tests pass, then **Submit** passes the hidden tests. Those compare your output with
scikit-learn's `RobustScaler` fitted on the training rows. The families they cover: random matrices
of several shapes; the real `digits` pixels, where a column can vary while its IQR is zero; a single
extreme value in the training rows, which leaves the center and scale unchanged while the mean and
standard deviation would move; held-out values that never change the fitted statistics; constant
and zero-IQR columns (scale one, including varying columns with zero IQR); an IQR of `1e-15`, which
gets scale one, while `3e-15` is kept; even and odd training counts for the median; integer inputs;
the empty-values case; invalid inputs; and inputs left unchanged.

# Covariance matrix

Implement `covariance_matrix(X, ddof=1)`, which returns the sample covariance matrix of a data
matrix. Each row of `X` is one observation and each column is one variable. Entry `[j, k]` of the
result is the average co-movement of variables `j` and `k` around their means, with the divisor
`n - ddof`.

## Contract

- `X`: 2-D array-like of real numbers, shape `(n, d)` with $n \ge 1$ rows and $d \ge 1$ columns.
  Integers allowed. Every entry is finite.
- `ddof`: a non-negative integer. The default `1` gives the unbiased sample covariance.
- Returns: NumPy array of shape `(d, d)` and dtype `float64`, equal to
  $C = \dfrac{(X - \bar{x})^\top (X - \bar{x})}{n - \text{ddof}}$, where $\bar{x}$ is the vector of column
  means.
- Raises `ValueError` when `X` is not 2-D, when it has no rows or no columns, when it contains NaN or
  infinity, when `ddof` is negative or not an integer, or when `n <= ddof` (there is no positive divisor).
- Must not mutate its input.

## Examples

```python
>>> covariance_matrix([[1, 2], [3, 4], [5, 9]])          # ddof=1, the default
array([[ 4.,  7.],
       [ 7., 13.]])
>>> covariance_matrix([[1, 2], [3, 4], [5, 9]], ddof=0)  # divide by n
array([[2.6667, 4.6667],
       [4.6667, 8.6667]])
>>> covariance_matrix([[1.0, 2.0]])                      # one row: n - ddof = 0
ValueError: need more observations than ddof: have 1, ddof is 1
```

## Constraints

- Allowed: NumPy array operations (`np.asarray`, `.mean`, `.T`, `@`, `np.isfinite`, `np.einsum`, ...).
- Not allowed: `numpy.cov`, `numpy.corrcoef`, `pandas` and `scipy`. Build the centred products yourself.

## Done looks like

All public tests pass, then **Submit** passes the hidden tests. Those check agreement with
`numpy.cov` on random data for several `ddof` values; symmetry and positive semi-definiteness;
invariance to shifting and the $s^2$ scaling rule; a large common offset (about $10^9$) where the
shortcut $\mathbb{E}[xy]-\mathbb{E}[x]\,\mathbb{E}[y]$ loses all precision; the single-variable shape;
the `ddof` errors (negative and fractional) and size errors; NaN, infinite, non-2-D, empty and
zero-variable input; integer input; and inputs left unchanged.

# Fit a standard scaler on training data

Implement `standardize(train, values)`. Learn one mean and one population
standard deviation per training feature, then transform `values` using them.

## Contract

- `train`: real numeric array-like of shape `(n, d)`, with `n >= 1`, `d >= 1`.
- `values`: real numeric array-like of shape `(m, d)`, with `m >= 0`.
- Entries must be finite and have absolute value at most `1e100`, inclusive, in both `train` and
  `values`.
- Returns a **tuple** `(transformed, mean, scale)` of float64 NumPy arrays with
  shapes `(m, d)`, `(d,)`, `(d,)`.
- Compute population scales with denominator `n` (`ddof=0`). An exactly constant
  training column (all its entries equal, compared exactly) uses scale `1.0`, and its mean is that value. Do not replace small *nonzero* scales by one.
- Compute the scale without underflow. Training values `-t` and `t`, where `t` is the smallest
  normal float64, have the scale `t`, not `0`.
- Transform with `(values - mean) / scale`. Never fit statistics on `values`.
- Raise `ValueError` for wrong ranks, empty training dimensions, mismatched
  feature counts, nonfinite entries or entries outside the magnitude bound.
- Must not mutate either input; read-only inputs are supported.
- For extreme scale ratios, mathematically unrepresentable float64 outputs may
  be infinite. Tests of numerical accuracy use representable results.

## Examples

```python
standardize([[1, 10], [3, 14]], [[5, 8]])
# (array([[3., -2.]]), array([2., 12.]), array([1., 2.]))
```

```python
standardize([[4, 1], [4, 3]], [[7, 2]])
# (array([[3., 0.]]), array([4., 2.]), array([1., 1.]))
```

## Constraints

Allowed: Python and NumPy. Not allowed: `sklearn`.
Aim for `O((n + m) * d)` time. Use float64 arithmetic. Training and transform
row counts need not match; only their feature counts must match.

## Done looks like

Your scaler matches an independent implementation on randomized matrices,
preserves shapes and inputs, handles constant and tiny-scale columns, and uses
training statistics even when held-out data has a very different distribution.
Invalid-input families are ranks, dimensions, feature counts, range and finiteness.

The hidden tests draw their bulk inputs from the real dataset `wine` (see `docs/DATASETS.md`).

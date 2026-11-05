# Rolling mean and standard deviation

Implement `rolling_mean_std(x, window, *, min_periods=None)`. It returns two arrays with one entry per
value of `x`: the trailing rolling mean and the trailing sample standard deviation (ddof = 1). The entry at
position `i` uses only `x[i - window + 1 .. i]`, so it never reads a later value. Update the statistics as
the window moves: add the newest value, remove the one that leaves, and do not recompute each window from
scratch.

## Contract

- `x`: 1-D sequence of finite real numbers (list, tuple or NumPy array), length n >= 0. Integers are allowed.
  Must not be mutated.
- `window`: integer >= 1. A bool or a float is rejected.
- `min_periods`: integer with 1 <= min_periods <= window, or `None`, which means `window`.
- Returns: `(means, stds)`, two float64 arrays of length n. An entry is NaN wherever the window holds fewer
  than `min_periods` values. The standard deviation is also NaN wherever the window holds one value.
- Raises `ValueError` when `x` is not 1-D, when it contains NaN or infinity, when `window` or `min_periods`
  is not a positive integer, or when `min_periods > window`.

## Examples

```python
>>> rolling_mean_std([1, 2, 3, 4, 5], 3)
(array([nan, nan,  2.,  3.,  4.]), array([nan, nan, 1., 1., 1.]))
```

```python
>>> rolling_mean_std([1, 2, 3, 4, 5], 3, min_periods=1)
(array([1. , 1.5, 2. , 3. , 4. ]), array([   nan, 0.70710678, 1.        , 1.        , 1.        ]))
```

The second example keeps the partial windows at the start. The first window holds one value, so its
standard deviation is NaN.

## Constraints

- Allowed: NumPy and the Python standard library (`math`, `numbers`, ...).
- Not allowed: `pandas` (the tests use its rolling methods as the reference, so the rolling update must be
  yours), and `scipy`.

## Done looks like

All public tests pass, then **Submit** passes the hidden tests. Those compare the output with pandas'
rolling mean and standard deviation on random series, across window and `min_periods` combinations
(including a window of one, partial windows, and a window longer than the series). They check the
variance denominator on a window of two, where the answer is known in closed form. They change the values
after a position and require the earlier outputs to stay the same, so no window may look forward. A long
series at a large common offset must keep its spread against a per-window two-pass computation. A constant
series must have zero spread. Readings near `1e200` and `1e-200`, whose squares overflow or underflow,
must keep their spread. Equal neighbours at window two must have a spread of zero up to rounding. The hidden tests also cover invalid inputs, empty input, integer input, and
inputs that must not be mutated.

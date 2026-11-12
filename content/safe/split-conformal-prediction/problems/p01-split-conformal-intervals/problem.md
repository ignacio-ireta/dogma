# Split conformal intervals for regression

Implement `split_conformal_interval(cal_pred, cal_true, test_pred, alpha)`. It returns the lower and
upper bounds of a split conformal prediction interval for each test point. The width comes from the
calibration residuals. Each test point gets the interval `[test_pred - q, test_pred + q]`, where `q` is
the conformal threshold at coverage level `1 - alpha`.

## Contract

- `cal_pred`, `cal_true`: real numeric array-likes, 1-D, the same nonzero length `n`, finite values.
  They are the calibration predictions and the true values.
- `test_pred`: real numeric array-like, 1-D, finite, length `m >= 0`.
- `alpha`: Python float, finite, `0 < alpha < 1`. A bool is invalid.
- Scores: `s_i = |cal_true[i] - cal_pred[i]|`.
- Rank: `k = ceil((n + 1) * (1 - alpha))`. Compute it exactly: the float product can be off by a
  tiny amount, and `(9 + 1) * (1 - 0.7)` rounds to `3.0000000000000004`, which would give rank 4
  instead of 3. Two ways that pass: `Fraction(repr(alpha))` (reads `0.7` as `7/10`, the decimal you
  wrote), or subtracting a tiny tolerance such as `1e-9` before `ceil`. `Fraction(alpha)` on the exact
  binary value is not accepted, because the float `0.3` is slightly below `3/10`.
- Threshold: `q` is the `k`-th smallest score (1-indexed). If `k > n`, then `q = +inf`.
- Returns: `(lower, upper)`, two float64 arrays of length `m`. `lower = test_pred - q` and
  `upper = test_pred + q`. When `q = +inf`, `lower` is `-inf` and `upper` is `+inf`.
- Raises `ValueError` for: an empty calibration set; mismatched lengths; non-1-D inputs; non-numeric
  or complex dtype; NaN or infinite values; `alpha` not a float in `(0, 1)`; a bool `alpha`.
- Must not mutate inputs.

## Examples

```python
>>> cal_pred = [0, 0, 0, 0, 0, 0, 0, 0, 0]
>>> cal_true = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.9]
>>> split_conformal_interval(cal_pred, cal_true, [5.0], 0.2)
# residuals are 0.1 .. 0.9; k = ceil(10 * 0.8) = 8; q = 0.8
(array([4.2]), array([5.8]))
```

```python
>>> split_conformal_interval(cal_pred, cal_true, [5.0], 0.05)
# k = ceil(10 * 0.95) = 10 > 9, so q = +inf
(array([-inf]), array([inf]))
```

## Constraints

- Allowed: NumPy and the Python standard library, including `math` and `fractions.Fraction`.
  Sort and index the scores yourself.
- Not allowed: `numpy.quantile`, `numpy.percentile`, `numpy.nanquantile` and `numpy.nanpercentile`.
  Each one computes the threshold for you, and the rank rule is the point of the exercise.

## Done looks like

Passing means the two examples match and the rank rule holds at every level. Hidden tests cover:

- seeded random calibration sets compared with an independent sorted-index computation;
- a rank that lands exactly on an integer, where float arithmetic would give an infinite interval;
- an infinite threshold when the rank exceeds `n`, and a single calibration point;
- tied residuals and an empty test set;
- scale invariance at very large and very small magnitudes;
- invariance to the order of calibration points;
- invalid inputs and integer inputs;
- a seeded simulation whose empirical coverage must be at least `1 - alpha`, within a stated bound.

The hidden tests draw their bulk inputs from the real dataset `california_housing` (see
`docs/DATASETS.md`).

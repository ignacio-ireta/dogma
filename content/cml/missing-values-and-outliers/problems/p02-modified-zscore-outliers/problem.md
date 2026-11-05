# Modified z-score outliers

Implement `robust_zscore_outliers(x, threshold=3.5)`. It flags outliers in a one-dimensional sample with the
modified z-score of Iglewicz and Hoaglin. The classic z-score uses the mean and the standard deviation, and
one large value inflates both, so it can hide itself. The modified score centers on the median and scales
by the median absolute deviation (MAD), which the outlier cannot move much. The formula, the strict comparison
and the zero-MAD convention are part of the contract.

## Contract

- `x`: one-dimensional array-like of finite real numbers, with at least one value. Integers are allowed.
- `threshold`: a positive finite number, default `3.5`.
- Let `median` be the median of `x` (the mean of the two middle values for an even count), and `MAD` the
  median of `|x - median|`. Compute each midpoint without overflow when its exact value is finite: the
  median of `1.6e308` and `1.7e308` is `1.65e308`.
- The modified z-score is `M = 0.6745 * (x - median) / MAD`, computed element by element. The
  difference `x - median` can overflow for finite values such as `-1.7e308` and `1.5e308`. `M` is still
  representable there, and it must be correct.
- Returns a boolean array of the same length as `x`: `True` where `|M| > threshold` (strictly greater), `False`
  elsewhere.
- **Zero MAD.** When `MAD` is exactly `0`, the score is undefined. In that case a value is flagged exactly when
  it differs from `median`. A value equal to the median is never flagged.
- Raises `ValueError` when `x` is not one-dimensional, when it is empty, when it holds a NaN or an infinity,
  or when `threshold` is not a positive finite number.
- Must not mutate `x`.

## Examples

```python
>>> robust_zscore_outliers([1, 2, 3, 4, 100])
array([False, False, False, False,  True])
    # median 3, MAD 1: M = 0.6745 * (x - 3), so only 100 has |M| > 3.5 (M = 65.4)
    # the classic z-score of 100 is about 2.0, below 3, so it would not be flagged
```

```python
>>> robust_zscore_outliers([5, 5, 5, 5, 9])
array([False, False, False, False,  True])     # MAD is 0: flag values that differ from the median 5
```

```python
>>> robust_zscore_outliers([1, 2, 3], threshold=0)
ValueError                                     # threshold must be positive
```

## Constraints

- Allowed: NumPy and the standard library.
- Not allowed: `scipy` and `statistics`. Compute the median and the MAD yourself. NumPy's `median` is allowed, subject to the overflow rule in
the Contract.

## Done looks like

All public tests pass, then **Submit** passes the hidden tests. Those compare your flags with an independent
computation, using SciPy's `median_abs_deviation`, on random samples with and without outliers, and over several
thresholds. They also check: that a single large value is flagged where the classic z-score would hide it; the
zero-MAD convention, including a sample where every value equals the median; invariance to shifting and to
positive scaling of `x`; symmetry under negation; the output shape and dtype; a monotone relation between the
threshold and the flagged set; integer input; large magnitudes; a single value; a comparison that is strict; and
the invalid cases (wrong rank, empty, NaN, infinity, and non-positive or non-finite thresholds). Inputs are left
unchanged.

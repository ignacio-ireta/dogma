# Conformal prediction sets for classification

Implement `prediction_sets(cal_probs, cal_labels, test_probs, alpha)`. It returns a boolean inclusion
mask: for each test row, which classes belong to the conformal prediction set at coverage level
`1 - alpha`. Each calibration point gets the score `1 - p(true class)`. The threshold is the conformal
quantile of those scores. Class `j` is in the set for test row `i` when `1 - test_probs[i, j] <= q`.

## Contract

- `cal_probs`: real numeric array-like, shape `(n, K)` with `n >= 1` and `K >= 2`. Entries are finite
  and nonnegative, and each row sums to 1 within `1e-9`.
- `cal_labels`: integer array of shape `(n,)` (dtype kind `i` or `u`). Each label is in `[0, K)`. Bool
  and float labels are invalid.
- `test_probs`: same rules as `cal_probs`, with shape `(m, K)`, `m >= 0`, and the same `K`.
- `alpha`: Python float, finite, `0 < alpha < 1`. A bool is invalid.
- Score: `s_i = 1.0 - cal_probs[i, cal_labels[i]]`, in float64.
- Rank: `k = ceil((n + 1)(1 - alpha))`. Compute it exactly: a float product can be off by a tiny amount,
  and `(9 + 1) * (1 - 0.7)` rounds to `3.0000000000000004`. Two ways that pass: `Fraction(repr(alpha))`
  (reads `0.7` as `7/10`, the decimal you wrote), or subtracting a tiny tolerance such as `1e-9` before
  `ceil`. `Fraction(alpha)` on the exact binary value is not accepted.
- Threshold: `q` is the `k`-th smallest score (1-indexed), or `+inf` when `k > n`.
- Returns: a `bool` array of shape `(m, K)`. `out[i, j]` is True when `(1.0 - test_probs[i, j]) <= q`.
  Equality counts as inside. A row of all False is an empty set and is allowed.
- Raises `ValueError` for any invalid argument above, including a class-count mismatch.
- Must not mutate inputs.

## Examples

```python
>>> cal_probs = [[0.7, 0.2, 0.1], [0.1, 0.6, 0.3], [0.2, 0.3, 0.5], [0.5, 0.4, 0.1]]
>>> cal_labels = [0, 1, 2, 0]
# scores are 0.3, 0.4, 0.5, 0.5
>>> prediction_sets(cal_probs, cal_labels, [[0.7, 0.2, 0.1]], 0.6)
# k = ceil(5 * 0.4) = 2, so q = 0.4; scores of the row are 0.3, 0.8, 0.9
array([[ True, False, False]])
```

```python
>>> prediction_sets(cal_probs, cal_labels, [[0.45, 0.45, 0.1], [0.5, 0.4, 0.1]], 0.2)
# k = ceil(5 * 0.8) = 4, so q = 0.5
array([[False, False, False],
       [ True, False, False]])
# row 2 scores are 0.5, 0.6, 0.9: 0.5 equals q, which is inside
```

## Constraints

- Allowed: NumPy and the Python standard library, including `math` and `fractions.Fraction`.
  Sort the scores yourself.
- Not allowed: `numpy.quantile`, `numpy.percentile`, `numpy.nanquantile` and `numpy.nanpercentile`.
  They compute the threshold for you, and the rank rule is the point of the exercise.

## Done looks like

Passing means both examples match and the rank rule holds at every level. Hidden tests cover:

- seeded random probabilities and labels compared with an independent sorted-index oracle;
- a rank that lands exactly on an integer, where a float product would make the threshold infinite;
- an infinite threshold that includes every class, and a confident row that always keeps its class;
- ties at the threshold, and empty sets;
- nested sets as alpha decreases;
- invalid probabilities, labels, class counts and alpha;
- output dtype and shape, including zero test rows;
- a seeded simulation whose empirical coverage must be at least `1 - alpha`, within a stated bound.

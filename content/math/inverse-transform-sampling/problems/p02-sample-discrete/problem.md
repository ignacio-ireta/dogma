# Discrete samples by inverse transform

Implement `sample_discrete(probs, n, rng)`. It returns `n` category indices drawn from the
distribution `probs` over `0, 1, ..., k-1`, by inverting the cumulative distribution. The cumulative
sums split `[0, 1)` into one stretch per category, and each uniform draw `u` selects the stretch it
falls in.

## Contract

- `probs`: a 1-D sequence of `k >= 1` finite, non-negative numbers. Their sum must be 1 up to
  floating-point rounding: `abs(sum - 1) <= 1e-9`.
- `n`: an int, `n >= 0`.
- `rng`: a NumPy `Generator`. The function makes exactly one call, `u = rng.random(n)`, and no
  other draw. (In tests `rng` may be any object with a `random(n)` method that returns `n` values in
  `[0, 1)`.)
- Let `c = cumsum(probs)` and `c_-1 = 0`. Category `i` is returned when `c[i-1] <= u < c[i]`.
  A draw `u` that is at least `c[k-1]` is returned as category `k-1`. That case happens when
  rounding leaves `c[k-1]` just below 1.
- Returns: an integer array of shape `(n,)`.
- Raises `ValueError` when `probs` is not a non-empty 1-D sequence, when an entry is negative or not
  finite, when the sum is not 1 within `1e-9`, or when `n` is negative.
- Must not mutate `probs`.

## Examples

```python
>>> sample_discrete([0.2, 0.3, 0.5], 3, np.random.default_rng(0))
array([2, 1, 0])      # u = [0.637, 0.270, 0.041] against c = [0.2, 0.5, 1.0]
>>> sample_discrete([0.2, 0.3, 0.5], 0, np.random.default_rng(0))
array([], dtype=int64)
>>> sample_discrete([0.2, 0.3, 0.4], 2, np.random.default_rng(0))
ValueError            # the probabilities sum to 0.9
```

## Constraints

- Allowed: NumPy (`np.cumsum`, `np.searchsorted`, `np.asarray`, …) and the `rng` you are given.
- Not allowed: `scipy`, and NumPy's or the standard library's categorical samplers, such as
  `rng.choice` with `p=`. The point is the inverse map applied to uniforms.

## Done looks like

All public tests pass, then **Submit** runs the hidden tests. They check: exact agreement with a
linear scan over the cumulative sums, for the same uniforms and for the seeded generator, with
zero-probability categories in the mix; the half-open rule at exact cumulative values; a zero-probability
category that is never returned; a rounding case at the top of the CDF that must map to the last
category; probabilities that sum to one only up to rounding, which must be accepted; invalid
probabilities (including sums just outside the tolerance) and `n` that must raise `ValueError`; one call to `rng.random(n)`; the integer dtype
and shape; a chi-square goodness-of-fit at a fixed seed; and no mutation of the input.

The hidden tests draw their bulk inputs from the real dataset `online_retail` (see
`docs/DATASETS.md`).

# Bias-variance from predictions

Implement `bias_variance(predictions, f_true, noise_var)`. Suppose you trained `B` models, each on its own training
set drawn from the same process, and each predicts the same `m` test points. The function measures how much of the
expected test error is squared bias, how much is variance, and how much is irreducible noise. The noise has mean zero and is independent of all fitted predictions. The decomposition is exact
for a uniformly sampled row of this supplied collection; it estimates population components from finitely many fits.
It works on the predictions alone: you do not need to know how the training sets were made.

## Contract

- `predictions`: 2-D array-like of shape `(B, m)`, with `B >= 1` and `m >= 1`. Row `b` holds model `b`'s predictions
  at the `m` test points. All values are finite.
- `f_true`: 1-D array-like of length `m`, finite. The noise-free target at each test point.
- `noise_var`: a finite, non-negative number. The variance of the label noise, so a test label is `f_true + noise`.
- For each test point `j`, with `pbar_j` the mean of `predictions[:, j]` over the `B` models:
  - squared bias: `bias2_j = (pbar_j - f_true[j]) ** 2`
  - variance: `var_j` = the mean over the models of `(predictions[b, j] - pbar_j) ** 2`, divisor `B`, not `B - 1`
- Returns `(bias2, variance, expected_mse)`, each averaged over the `m` test points:
  - `bias2` is the mean of `bias2_j`
  - `variance` is the mean of `var_j`
  - `expected_mse` is the mean of `bias2_j + var_j + noise_var`, which equals `bias2 + variance + noise_var`
- Returns Python floats. Use float64 arithmetic, preserving representable errors even when the
  predictions share a very large common offset. Squared quantities genuinely beyond float64 range
  may return positive infinity; underflow below the smallest float64 subnormal may return zero.
- Raises `ValueError` when `predictions` is not 2-D or has no model or no test point; when `f_true` is not 1-D or its
  length is not `m`; when any value in `predictions` or `f_true` is not finite; or when `noise_var` is negative or not
  finite.
- Must not mutate inputs.

## Examples

```python
>>> bias_variance([[1.0, 2.0], [3.0, 2.0]], [2.0, 2.0], 0.5)
(0.0, 0.5, 1.0)         # the models average to the truth; their spread is 1 at point 0 and 0 at point 1
>>> bias_variance([[1.0], [1.0], [1.0]], [3.0], 0.0)
(4.0, 0.0, 4.0)         # three identical models, each 2 below the truth: all error is bias
>>> bias_variance([[1.0, 2.0]], [1.0, 2.0], 0.0)
(0.0, 0.0, 0.0)         # one model that is exactly right
```

## Constraints

- Allowed: NumPy and the standard library.

## Done looks like

All public tests pass, then **Submit** passes the hidden tests. They compare the three outputs with an independent
direct formula, check the identity `expected_mse = bias2 + variance + noise_var`, and check constant and single-model
edge cases. They run a fixed, seeded simulation of polynomial fits and check the expected trend for that particular
setup; this is not a guarantee that every learner has monotone bias or variance as capacity rises. The hidden tests also check the shape and value rules, the scaling behaviour when
predictions, truth and noise are scaled together, the return types, and that the inputs are left unchanged.

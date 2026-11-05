# Permutation feature importance

Permutation importance measures how much a fitted model relies on each feature. Shuffle one column of held-out data, which breaks its link to the target while keeping its distribution, and record how much the score drops. Implement `permutation_importance(predict, X, y, metric, *, n_repeats, seed)`, which returns the mean and the standard deviation of the score drop for each feature.

## Contract
- `predict`: a callable that takes a 2-D float array of shape `(n, d)` and returns `n` predictions, as a 1-D array-like.
- `X`: 2-D array-like of finite numbers, shape `(n, d)`, with $n \ge 1$ and $d \ge 1$.
- `y`: 1-D array-like of length `n`.
- `metric`: a callable `metric(y_true, y_pred)` that returns a number. Larger is better, as for accuracy or $R^2$.
- `n_repeats`: keyword-only integer, at least 1. `seed`: keyword-only non-negative integer.
- **Protocol (exact, so results are reproducible):**
  1. `baseline = metric(y, predict(X))`.
  2. Create one generator, `rng = np.random.default_rng(seed)`, and use it for everything that follows.
  3. For each feature `j` in order `0, 1, ..., d-1`, and for each repeat `r` in order `0, ..., n_repeats-1`: draw `perm = rng.permutation(n)`, copy `X`, replace column `j` with `X[perm, j]` and leave the other columns as they are. Record `drop[j, r] = baseline - metric(y, predict(X_perm))`.
  4. `mean_drop[j]` is the mean of `drop[j, :]`. `std_drop[j]` is the **population** standard deviation of `drop[j, :]` (divide by $n_{\text{repeats}}$, not by $n_{\text{repeats}}-1$).
- Returns: `(mean_drop, std_drop)`, two `float64` arrays of shape `(d,)`. A positive mean means the feature helps the model.
- Raises: `ValueError` when `X` is not a non-empty 2-D finite array; when `y` is not 1-D or has a different length from `X`; when `n_repeats` is not an integer of at least 1; when `seed` is not a non-negative integer; and when `predict` does not return one prediction per row.
- Must not mutate `X` or `y`.

## Examples

```python
>>> import numpy as np
>>> X = np.array([[0.0, 5.0], [1.0, 3.0], [2.0, 9.0], [3.0, 1.0]])
>>> y = np.array([0, 0, 1, 1])
>>> def predict(Z):
...     return (Z[:, 0] >= 1.5).astype(int)  # uses column 0 only
>>> def accuracy(y_true, y_pred):
...     return float(np.mean(y_true == y_pred))
>>> mean_drop, std_drop = permutation_importance(predict, X, y, accuracy, n_repeats=3, seed=0)
>>> float(mean_drop[1]), float(std_drop[1])
(0.0, 0.0)
>>> _, std_one = permutation_importance(predict, X, y, accuracy, n_repeats=1, seed=0)
>>> std_one
array([0., 0.])
```

## Constraints

- Allowed: NumPy and the standard library.
- Not allowed: `sklearn` (including `sklearn.inspection.permutation_importance`) and `scipy`. Their random streams are not the protocol above.

## Done looks like

All public tests pass, then **Submit** runs the hidden tests. They check: exact agreement with an independent loop that follows the protocol, including the order in which draws are made and the population standard deviation; a feature the model ignores has exactly zero mean and standard deviation; a single repeat has zero spread; a feature the model uses has a positive drop; the same seed gives the same result and a different seed gives a different one; invalid `n_repeats`, `seed`, shapes, lengths and a wrong-length prediction raise `ValueError`; and `X` and `y` are left unchanged.

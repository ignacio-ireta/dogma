# Cross-validated score

Implement `cross_validate(fit_predict, X, y, folds, metric)`. It evaluates a model over a given list of
folds. For each fold, it fits on the training rows, predicts the test rows, and scores the predictions.
It returns the per-fold scores and their mean and standard deviation. The model is supplied as a function,
so the splitting and the scoring stay separate from any particular library.

## Contract

- `fit_predict(X_train, y_train, X_test)`: a callable. It receives the training rows and targets and the
  test rows, and returns a 1-D array-like with one prediction per test row. It is called once per fold, in
  fold order.
- `X`: array-like with `n` rows (first axis = samples). `X` is indexed by row.
- `y`: 1-D array-like of length `n`.
- `folds`: a sequence of at least two `(train_idx, test_idx)` pairs. Each index array is a non-empty 1-D
  array of integers in `0 .. n - 1`. Folds are used exactly as given: they need not be a partition, and
  they need not be contiguous or equal in size.
- `metric(y_true, y_pred)`: a callable returning one float. Its first argument is the test targets of the
  fold, its second is that fold's predictions. The argument order is fixed: targets first.
- Returns: a tuple `(scores, mean, std)`.
  - `scores`: a list of floats, one per fold, in fold order.
  - `mean`: the unweighted mean of `scores`, a float.
  - `std`: the sample standard deviation of `scores` with one degree of freedom (`ddof = 1`), a float.
- Raises `ValueError` when `len(X) != len(y)`, when `y` is not 1-D, when fewer than two folds are given
  (a standard deviation needs two scores), when an index array is empty, non-integer, or out of range, or
  when `fit_predict` returns a number of predictions different from the number of test rows.
- Must not mutate inputs.

## Examples

```python
>>> def mean_model(X_tr, y_tr, X_te):          # predicts the training mean for every test row
...     return [float(np.mean(y_tr))] * len(X_te)
>>> mae = lambda t, p: float(np.mean(np.abs(t - p)))
>>> folds = [(np.array([2, 3]), np.array([0, 1])), (np.array([0, 1]), np.array([2, 3]))]
>>> cross_validate(mean_model, [[0.], [0.], [0.], [0.]], [1., 1., 3., 3.], folds, mae)
([2.0, 2.0], 2.0, 0.0)
```

```python
>>> three = [(np.array([1, 2]), np.array([0])), (np.array([0, 2]), np.array([1])),
...          (np.array([0, 1]), np.array([2]))]
>>> scores = iter([1.0, 2.0, 3.0])              # the metric returns these in fold order
>>> cross_validate(mean_model, [[0.], [0.], [0.]], [1., 2., 3.], three, lambda t, p: next(scores))
([1.0, 2.0, 3.0], 2.0, 1.0)
```

```python
>>> cross_validate(mean_model, [[0.], [0.], [0.], [0.]], [1., 1., 3., 3.], folds[:1], mae)
ValueError                  # one fold gives no standard deviation
```

## Constraints

- Allowed: NumPy (indexing, `np.asarray`, `np.mean`, `np.std`, `np.issubdtype`, …) and the standard library.
- Not allowed: `sklearn`. It provides `cross_val_score`, which would solve the task for you.

## Done looks like

All public tests pass, then **Submit** passes the hidden tests. They compare the per-fold scores, mean and
standard deviation with scikit-learn's `cross_val_score` on the same folds, for least-squares and
asymmetric metrics, with folds that are unequal in size and not contiguous. They check the standard
deviation's degrees of freedom, the argument order of `metric`, that `fit_predict` never sees a test row
and is called once per fold, the validation rules, determinism, and that the inputs are left unchanged.

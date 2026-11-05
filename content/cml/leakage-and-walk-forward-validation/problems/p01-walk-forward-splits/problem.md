# Walk-forward splits

Implement `walk_forward_splits(n, *, initial_train, horizon, step, gap=0, expanding=True)`. It produces
the train/test splits for evaluating a model on a time-ordered series of `n` rows. Each split trains on the
past and tests on the future, and the test window moves forward by `step` rows at a time. Random folds would
let the model train on rows that come after its test rows, so this is the split to use for forecasting.

## Contract

- `n`: int, the number of rows, in time order (row `0` is the oldest).
- `initial_train`: int `>= 1`. The size of the first training window. With `expanding=True`, later windows
  grow from row 0. With `expanding=False`, every window has exactly `initial_train` rows.
- `horizon`: int `>= 1`. The number of rows in each test window.
- `step`: int `>= 1`. How far the training cutoff moves between consecutive splits.
- `gap`: int `>= 0`. The number of rows skipped between the last training row and the first test row,
  for example because labels arrive late.
- `expanding`: bool, default `True`.
- **Split `j`** (`j = 0, 1, 2, …`) has cutoff `c_j = initial_train + j * step`. Its training rows are
  `[0, c_j)` when expanding, or `[c_j - initial_train, c_j)` when rolling. Its test rows are
  `[c_j + gap, c_j + gap + horizon)`.
- Splits are generated while the test window ends at or before `n`: `c_j + gap + horizon <= n`. The first
  split is the same in both modes.
- Returns: a list of `(train_idx, test_idx)` pairs in split order. Both arrays are 1-D `np.intp`, consecutive
  and ascending.
- Raises `ValueError` when `initial_train`, `horizon` or `step` is below 1, when `gap` is negative, or when
  `n < initial_train + gap + horizon` (no split fits).
- Must not mutate inputs.

## Examples

```python
>>> [(tr.tolist(), te.tolist()) for tr, te in walk_forward_splits(10, initial_train=4, horizon=2, step=2)]
[([0, 1, 2, 3], [4, 5]), ([0, 1, 2, 3, 4, 5], [6, 7]), ([0, 1, 2, 3, 4, 5, 6, 7], [8, 9])]
>>> [(tr.tolist(), te.tolist()) for tr, te in walk_forward_splits(10, initial_train=4, horizon=2, step=2, gap=1)]
[([0, 1, 2, 3], [5, 6]), ([0, 1, 2, 3, 4, 5], [7, 8])]
>>> [(tr.tolist(), te.tolist()) for tr, te in walk_forward_splits(10, initial_train=4, horizon=2, step=2, expanding=False)]
[([0, 1, 2, 3], [4, 5]), ([2, 3, 4, 5], [6, 7]), ([4, 5, 6, 7], [8, 9])]
>>> walk_forward_splits(3, initial_train=4, horizon=1, step=1)
ValueError                  # not even one split fits
```

## Constraints

- Allowed: NumPy (`np.arange`, `np.intp`, …) and the standard library.
- Not allowed: `sklearn`, which provides `TimeSeriesSplit` and would solve the task for you.

## Done looks like

All public tests pass, then **Submit** passes the hidden tests. They compare the splits with an explicit
reference over several configurations (expanding and rolling, with and without a gap), and check the
leakage properties: every training row precedes every test row by more than the gap, the train and test
windows never overlap, and no test window runs past the last row. They also check that the windows advance by
exactly `step`, that no further split would fit, the window lengths, and that the output matches
scikit-learn's `TimeSeriesSplit` for one matching configuration. The hidden tests also reject invalid
parameters and check that index arrays are integer-typed and ascending.

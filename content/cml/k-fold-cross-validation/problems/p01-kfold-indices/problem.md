# K-fold split indices

Implement `kfold_indices(n, k, *, shuffle=False, seed=None)`. It splits the row positions `0` to `n - 1`
into `k` folds and returns, for each fold, the training positions and the test positions. Fold `j` is
the test set of split `j`. Its training set is every other position.

## Contract

- `n`: int, the number of samples.
- `k`: int, the number of folds. Must satisfy `2 <= k <= n`.
- `shuffle`: bool, default `False`.
- `seed`: int or `None`. Used only when `shuffle` is true.
- **Fold sizes.** Let `q, r = divmod(n, k)`. The first `r` folds have `q + 1` test positions and the
  remaining folds have `q`.
- **Unshuffled (`shuffle=False`).** Fold test sets are consecutive blocks of positions in increasing
  order. Fold 0 starts at position 0. `seed` is ignored.
- **Shuffled (`shuffle=True`).** `seed` must be an int. Let
  `perm = np.random.default_rng(seed).permutation(n)`. The blocks above, taken in order, become
  consecutive chunks of `perm`: fold `j`'s test set is `perm[start_j:stop_j]`. The shuffled folds are
  not guaranteed to match any library's shuffled folds. This protocol is the contract.
- Returns: a list of `k` pairs `(train_idx, test_idx)` in fold order. Both are 1-D `np.intp` arrays.
  Each array is sorted in ascending order. `train_idx` is the complement of `test_idx` in `range(n)`.
- Raises `ValueError` when `k < 2`, when `k > n`, or when `shuffle` is true and `seed` is `None`.
- Must not mutate inputs.

## Examples

```python
>>> [te.tolist() for _, te in kfold_indices(10, 3)]
[[0, 1, 2, 3], [4, 5, 6], [7, 8, 9]]
>>> [te.tolist() for _, te in kfold_indices(6, 3, shuffle=True, seed=0)]
[[2, 3], [4, 5], [0, 1]]
>>> kfold_indices(3, 4)
ValueError                  # there are more folds than samples
```

## Constraints

- Allowed: NumPy (`np.arange`, `np.random.default_rng`, `np.sort`, `np.setdiff1d`, `np.array_split`, …).
- Not allowed: `sklearn`, which provides `KFold` and would solve the task for you.

## Done looks like

All public tests pass, then **Submit** passes the hidden tests. The hidden tests compare unshuffled
folds with scikit-learn's `KFold` for several sizes, check shuffled folds against the seeded protocol,
check the partition property (each position is tested exactly once and the training set is its
complement) on random sizes, verify the remainder rule for fold sizes, cover leave-one-out (`k = n`),
reject invalid `k` and a missing seed, confirm that the same seed reproduces the folds and that
`seed` is ignored without shuffling, and check that every index array is sorted and integer-typed.

Some hidden tests take their inputs from the real datasets `breast_cancer`, `diabetes`, `iris` and
`wine` (see `docs/DATASETS.md`).

# Seeded train/test split with stratification

Implement `train_test_split_indices(n, test_size, *, seed, stratify=None)`. It returns the row
indices of a training set and a test set. Both come from one seeded random generator, so the
split is part of the contract: two correct implementations must return the same index arrays.
The protocol below says which random draws are made, in which order, and how the counts are
rounded.

## Contract

- `n`: integer, `n >= 2`.
- `test_size`: float strictly between 0 and 1.
- `seed`: integer, keyword-only. It is used only to build the random generator.
- `stratify`: `None`, or a 1-D array-like of length `n` with labels that can be sorted (integers
  or strings). Default `None`.
- Returns: `(train_idx, test_idx)`, two int64 arrays, each sorted in ascending order. Together
  they hold every row index `0..n-1` exactly once.
- Raises `ValueError` when `n < 2`; when `test_size` is not strictly between 0 and 1; when
  `stratify` is not 1-D with length `n`; when the training side would be empty; and, for a
  stratified split, when some class would have no training row.
- Must not mutate `stratify`.

**Protocol (exact).** Every product below is a float64 product evaluated as written, and `ceil` and
`floor` act on those floats. Let `n_test = ceil(test_size * n)`. The test side gets exactly `n_test`
rows. Float64 matters: `0.07 * 100` is `7.000000000000001`, so `n_test` is 8, which is also the count
scikit-learn gives. Exact decimal arithmetic would give 7, and that is wrong for this protocol.

*Without `stratify`:*

1. Create `rng = np.random.default_rng(seed)` once.
2. Draw `perm = rng.permutation(n)`. This is the only draw.
3. `test_idx` is the sorted `perm[:n_test]`, and `train_idx` is the sorted `perm[n_test:]`.

*With `stratify` (labels `y`):*

1. Create `rng = np.random.default_rng(seed)` once, before any draw.
2. The classes are the distinct labels in ascending order, `c_1 < ... < c_k`. Let `n_j` be the
   number of rows with label `c_j`.
3. Allocate test rows per class. Let `ideal_j = test_size * n_j` (float64) and `base_j = floor(ideal_j)`.
   Let `R = n_test - sum(base_j)`. The `R` classes with the largest fractional parts
   `ideal_j - base_j` each get one extra test row. Ties in the fractional part go to the class that
   comes first in ascending order. Call the result `t_j`.
4. If some class has `t_j >= n_j`, raise `ValueError`, before any draw.
5. For `j = 1..k` in ascending order, draw `perm_j = rng.permutation(rows_j)`, where `rows_j` is
   the ascending row indices of class `j`. That is exactly `k` draws.
6. The first `t_j` entries of `perm_j` are test rows; the rest are training rows.
7. `test_idx` and `train_idx` are the unions over all classes, each sorted ascending.

## Examples

```python
>>> train_test_split_indices(10, 0.25, seed=0)   # n_test = ceil(2.5) = 3, not round(2.5) = 2
(array([0, 1, 3, 5, 7, 8, 9]), array([2, 4, 6]))

>>> y = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]           # class 0 has 4 rows, class 1 has 6
>>> train_test_split_indices(10, 0.5, seed=0, stratify=y)   # ideal 2.0 and 3.0, so no extra rows
(array([1, 3, 4, 5, 8]), array([0, 2, 6, 7, 9]))

>>> len(train_test_split_indices(100, 0.07, seed=0)[1])   # 0.07 * 100 is 7.000000000000001, so 8 rows
8

>>> train_test_split_indices(2, 0.9, seed=0)     # n_test = ceil(1.8) = 2 leaves no training rows
ValueError
```

The stratified rule is this protocol, not scikit-learn's `StratifiedShuffleSplit`. That class computes
its shares with a different expression and breaks ties with random draws, so its rows will not match
this one.

## Constraints

- Allowed: NumPy, the `math` module, and `np.random.default_rng(seed)`, which the protocol needs.
- Not allowed: `random` (the standard library generator would draw different numbers), `sklearn`
  and `scipy`. The split is the point of the exercise, so write the allocation and the draws
  yourself.

## Done looks like

All public tests pass, then **Submit** passes the hidden tests. Those compare your split with an
independent, loop-by-loop statement of the protocol on random sizes, test sizes, seeds and label
sets, both stratified and not. They also check: the test count against the ceiling rule (including
sizes where rounding would give a different count, and float64 products that land just above an
integer); disjoint, complete and sorted outputs;
per-class test counts within one row of their ideal count; labels whose first-seen order differs
from their sorted order; a singleton class that stays in training; reproducibility for a fixed seed
and dependence on the seed; invalid `n`, `test_size`, and `stratify` shapes; empty training sides;
integer dtype; and the labels left unchanged.

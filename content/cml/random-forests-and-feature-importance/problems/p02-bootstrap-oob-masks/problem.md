# Bootstrap samples and out-of-bag masks

A random forest trains each tree on a bootstrap sample: $n$ rows drawn **with replacement** from the $n$ training rows. About a third of the rows are left out of each draw. Those are the tree's out-of-bag (OOB) rows, and they give a built-in validation set. Implement `bootstrap_oob(n, n_trees, seed)`, which returns the in-bag counts and the OOB masks for every tree.

## Contract
- `n`: integer, at least 1, the number of training rows.
- `n_trees`: integer, at least 1.
- `seed`: non-negative integer.
- **Protocol (exact):** create one generator, `rng = np.random.default_rng(seed)`. For tree `t = 0, ..., n_trees-1`, in order, draw `rng.integers(0, n, size=n)`. Those are the indices of the tree's bootstrap sample, with repeats. `in_bag[t, i]` is the number of times row `i` was drawn.
- Returns: `(in_bag, oob)`.
  - `in_bag`: integer array of shape `(n_trees, n)`. Each row sums to `n`.
  - `oob`: boolean array of the same shape, with `oob[t, i] = (in_bag[t, i] == 0)`.
- Raises: `ValueError` when `n` or `n_trees` is not an integer of at least 1, or when `seed` is not a non-negative integer.

## Examples

```python
>>> in_bag, oob = bootstrap_oob(4, 2, seed=0)
>>> in_bag.sum(axis=1).tolist()
[4, 4]
>>> oob.dtype.kind, in_bag.dtype.kind
('b', 'i')
>>> bootstrap_oob(1, 3, seed=7)[1].tolist()
[[False], [False], [False]]
```

## Constraints

- Allowed: NumPy.
- Not allowed: `sklearn` (including `sklearn.utils.resample`), `scipy`, and the standard-library `random` module. Draw every index from the NumPy generator described above.

## Done looks like

All public tests pass, then **Submit** runs the hidden tests. They check: exact agreement with an independent counter that follows the protocol; every row of the in-bag counts sums to `n`; the OOB mask is exactly the zero counts; the out-of-bag fraction is close to $1/e$ on a large sample, which catches sampling without replacement; different trees get different samples; a one-row training set is always in bag; the same seed reproduces the output; invalid `n`, `n_trees` and `seed` raise `ValueError`; and the dtypes are integer for counts and boolean for masks.

Some hidden tests take their inputs from the real datasets `digits`, `iris` and `wine` (see
`docs/DATASETS.md`).

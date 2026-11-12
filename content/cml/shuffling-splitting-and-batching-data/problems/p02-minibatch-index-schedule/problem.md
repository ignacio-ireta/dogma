# Mini-batch index schedule

Implement `minibatch_indices(n, batch_size, epochs, *, seed, drop_last=False)`. It returns the
row indices that a training loop would visit, batch by batch, for a number of epochs. The
shuffling is part of the contract: each epoch's order comes from one seeded generator, so two
correct implementations return the same index arrays. The stochastic gradient descent problem
uses the same protocol for its updates; this problem isolates the index bookkeeping.

## Contract

- `n`: integer, `n >= 1`.
- `batch_size`: integer with `1 <= batch_size <= n`.
- `epochs`: integer, `epochs >= 0`.
- `seed`: integer, keyword-only. It is used only to build the random generator.
- `drop_last`: bool, keyword-only, default `False`.
- Returns: a Python list with one entry per epoch. The entry for each epoch is a list of
  1-D int64 arrays, one per batch, in the order they are visited.
- Raises `ValueError` when `n < 1`, when `batch_size` is not an integer in `1..n`, or when
  `epochs` is negative.

**Protocol (exact).**

1. Create `rng = np.random.default_rng(seed)` once, before the first epoch.
2. For each epoch, in order, draw `perm = rng.permutation(n)`. That is exactly one draw per epoch.
3. The batches of the epoch are consecutive slices of `perm`: `perm[0:B]`, `perm[B:2B]`, ... where
   `B = batch_size`. Every batch keeps the order of `perm`; do not sort it.
4. The last slice may be shorter than `B`. If `drop_last` is `True`, drop it. If `drop_last` is
   `False`, keep it.
5. Epochs with `epochs = 0` give an empty list.

## Examples

```python
>>> minibatch_indices(5, 2, 2, seed=0)
[[array([2, 4]), array([3, 0]), array([1])],
 [array([4, 1]), array([2, 0]), array([3])]]
    # seed 0 gives the permutations [2, 4, 3, 0, 1] and then [4, 1, 2, 0, 3]

>>> minibatch_indices(5, 2, 1, seed=0, drop_last=True)
[[array([2, 4]), array([3, 0])]]     # the short final batch [1] is dropped

>>> minibatch_indices(5, 2, 0, seed=0)
[]

>>> minibatch_indices(5, 0, 1, seed=0)
ValueError                           # batch_size must be at least 1
```

## Constraints

- Allowed: NumPy and `np.random.default_rng(seed)`, which the protocol needs.
- Not allowed: `random` (the standard library generator draws different numbers), `sklearn` and
  `scipy`. Build the slices yourself.

## Done looks like

All public tests pass, then **Submit** passes the hidden tests. Those compare your schedule with
an independent loop-by-loop statement of the protocol on random sizes, batch sizes, epoch counts
and both settings of `drop_last`. They also check: that each epoch is a fresh permutation of one
generator, so epochs differ; that each row appears exactly once per epoch when the last batch is
kept; that `drop_last` removes only a short final batch and changes nothing when the batches divide
evenly; the single full batch case; that batches keep permutation order; seed reproducibility and
seed dependence; zero epochs; integer index dtype; and invalid `n`, `batch_size`, `epochs` and
types.

Some hidden tests take their inputs from the real datasets `iris` and `wine` (see
`docs/DATASETS.md`).

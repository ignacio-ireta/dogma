# Mini-batch gradient descent

Implement `minibatch_sgd(X, y, lr, batch_size, epochs, *, seed)`. It fits least squares without an
intercept, minimising L(w) = ½·mean((Xw − y)²), by mini-batch gradient descent. The shuffling is part
of the contract: the result depends on the order in which rows are visited, so the protocol below
says exactly how the rows are chosen. Two correct implementations then return the same numbers.

## Contract

- `X`: 2-D array-like, shape (n, d), n ≥ 1, finite real numbers. Integers allowed.
- `y`: 1-D array-like, shape (n,).
- `lr`: learning rate, a number > 0.
- `batch_size`: integer with 1 ≤ `batch_size` ≤ n.
- `epochs`: integer ≥ 0.
- `seed`: integer, keyword-only. It is used only to build the random generator.
- Returns: `w`, a float64 array of shape (d,), the weights after the last epoch.
- Raises `ValueError` when `X` is not 2-D; when `y` is not 1-D or its length differs from n; when
  `batch_size` is outside 1..n; when `lr` is not positive; when `epochs` is negative.
- Must not mutate `X` or `y`.

**Protocol (exact).**

1. Start with `w = zeros(d)`.
2. Create `rng = np.random.default_rng(seed)` once, before the first epoch.
3. Each epoch, draw `perm = rng.permutation(n)`: exactly one draw per epoch.
4. The batches are consecutive slices of `perm`: `perm[0:B]`, `perm[B:2B]`, …, where `B = batch_size`.
   The last slice may be smaller, and it is kept. Let `idx` be the rows of a slice and `m = |idx|`.
5. For each slice, compute the gradient `g = X[idx]ᵀ (X[idx] w − y[idx]) / m` and update
   `w ← w − lr · g`. One update per slice, so an epoch makes ⌈n/B⌉ updates.
6. After the last epoch, return `w`.

## Examples

```python
>>> X, y = [[1.], [2.], [3.], [4.]], [2., 4., 6., 8.]
>>> minibatch_sgd(X, y, lr=0.1, batch_size=4, epochs=1, seed=0)
array([1.5])     # one full batch: gradient -15, so w = 0 + 0.1 * 15
>>> minibatch_sgd(X, y, lr=0.1, batch_size=2, epochs=1, seed=0)
array([2.])      # seed 0 gives perm = [2, 0, 1, 3]: two updates of +1 each
>>> minibatch_sgd(X, y, lr=0.1, batch_size=3, epochs=1, seed=0)
array([2.64])    # slices [2, 0, 1] and [3]: the last slice has one row
>>> minibatch_sgd(X, y, lr=0.1, batch_size=0, epochs=1, seed=0)
ValueError       # batch_size must be at least 1
```

## Constraints

- Allowed: NumPy array operations and `np.random.default_rng(seed)`, which the protocol needs.
- Not allowed: `numpy.linalg`, `scipy` and `sklearn`. Do not solve the least-squares problem in
  closed form: the point is the iteration.

## Done looks like

All public tests pass, then **Submit** passes the hidden tests. Those check agreement with an
independent row-by-row implementation of the protocol, for batch sizes that leave a smaller last
slice and for pure single-sample steps; the special case where one batch holds all rows; the batch
gradient against finite differences of the batch loss; convergence to the least-squares fit on
noise-free data; seed reproducibility and seed dependence; linearity in the targets; validation of
batch size, shapes, learning rate and epochs; integer inputs; and inputs left unchanged.

The hidden tests draw their bulk inputs from the real dataset `diabetes` (see `docs/DATASETS.md`).

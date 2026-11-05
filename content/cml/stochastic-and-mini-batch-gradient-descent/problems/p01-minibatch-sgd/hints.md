## Hint 1 — concept

The gradient for a batch is the average of the per-row gradients in that batch, so a smaller batch
gives a noisier but cheaper step. Shuffling decides which rows share a batch, so it decides the
steps. The protocol fixes the order of the random draws, which is why the seed matters.

## Hint 2 — formula / approach

Build one generator with `default_rng(seed)` before the epoch loop. Each epoch, draw
`perm = rng.permutation(n)`, then walk `range(0, n, batch_size)` and take `perm[start:start+batch_size]`.
For each slice, the gradient is `X_b.T @ (X_b @ w - y_b) / len(rows)`, and the update is `w - lr * g`.
Divide by the slice's own length, not by `batch_size`, because the last slice can be shorter.

## Hint 3 — code shape

Validate first: `X_arr.ndim == 2`, `y_arr.shape == (n,)`, `1 <= batch_size <= n`, `lr > 0`, `epochs >= 0`.
Then `rng = np.random.default_rng(seed)`, `w = np.zeros(d)`, and nested loops:
`for _ in range(epochs): perm = rng.permutation(n)` and
`for start in range(0, n, batch_size): rows = perm[start : start + batch_size]`.
Return `w` after the loops.

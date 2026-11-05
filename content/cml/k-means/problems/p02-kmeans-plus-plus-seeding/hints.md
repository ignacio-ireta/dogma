## Hint 1 — concept

k-means++ spreads the starting centres out. A row far from every chosen centre is more likely to be picked next, but nothing is forced. Randomness comes from one generator, and the protocol fixes exactly how it is used, so the same seed always gives the same centres.

## Hint 2 — formula / approach

Keep an array `d2` with the squared distance from every row to its nearest chosen centre. Start it from the first centre, and update it with `np.minimum` each time a centre is added. Sample the next index with probability `d2 / d2.sum()`. Passing that probability to `rng.choice` directly follows the stated protocol. An alternative must consume equivalent generator draws and produce the same indices. A common power-of-two rescaling before computing distances preserves the sampling probabilities and avoids squaring very large or tiny units.

## Hint 3 — code shape

Validate first: `X` is a finite 2-D array, `k` is an integer in $[1, n]$, `seed` is a non-negative integer, and `np.unique(X, axis=0)` has at least `k` rows. Create `rng = np.random.default_rng(seed)`, draw `first = rng.integers(n)`, and set `d2` from that row. Loop $k - 1$ times: sample with `rng.choice(n, p=d2 / d2.sum())`, append the index, and take the element-wise minimum of `d2` with the new distances. Return `X[chosen]`.

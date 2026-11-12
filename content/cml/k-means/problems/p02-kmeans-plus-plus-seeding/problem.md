# k-means++ seeding

Lloyd's algorithm depends on its starting centroids. k-means++ chooses them one at a time, each new centre far from the existing ones with high probability. Implement `kmeans_plus_plus(X, k, *, seed)`, which returns the $k$ chosen centres as an array of shape `(k, d)`.

The procedure is exact, so every learner gets the same centres for the same seed. Checking is on the output, so any code that draws the same indices from the same generator is correct. `rng.choice` is the direct way to do it:

1. `rng = np.random.default_rng(seed)`. This is the only source of randomness.
2. **First centre.** Draw `first = rng.integers(n)`, an index in $\{0, \dots, n-1\}$, and take row `first` of `X` as centre 1.
3. **Next centres.** For $j = 2, \dots, k$, compute for every row $i$ the squared distance $D_i^2$ to its nearest centre chosen so far. Then draw `idx = rng.choice(n, p=D2 / D2.sum())`, and take row `idx` as centre $j$.
4. Return the centres in the order chosen.

Each new centre is therefore drawn with probability proportional to $D^2$. A row that is already a centre has $D^2 = 0$, so it is never drawn again.

## Contract
- `X`: 2-D array-like of finite numbers, shape `(n, d)`, with $n \ge 1$ and $d \ge 1$.
- `k`: integer, with $1 \le k \le n$. Booleans are not integers here.
- `seed`: a non-negative integer. It is the only argument of the random generator.
- `X` must have at least `k` distinct rows, so that $\sum_i D_i^2 > 0$ at every step.
- Returns: a `float64` array of shape `(k, d)`. Each row is a row of `X`, copied.
- Raises: `ValueError` for an invalid `k` or `seed`, bad shapes, non-finite values, or too few distinct rows.
- Scaling `X` by an exact power of two such as `2.0**600` or `2.0**-600` must pick the same rows, so $D_i^2$ must not overflow or underflow.
- Must not mutate inputs.

## Examples

```python
>>> X = [[0.0], [1.0], [3.0], [7.0]]
>>> kmeans_plus_plus(X, 2, seed=0)
array([[7.],
       [0.]])
>>> kmeans_plus_plus([[1.0], [1.0], [2.0]], 3, seed=0)
ValueError: X must have at least k distinct rows
>>> kmeans_plus_plus(X, 2, seed=-1)
ValueError: seed must be a non-negative integer
```

With seed 0 the generator first picks row 3 (value 7), then row 0 (value 0), which has the largest weight among the remaining rows. The second example has only two distinct values, so it cannot supply three distinct centres.

## Constraints
- Allowed: NumPy and the standard library (`math`). Use `np.random.default_rng` exactly as specified.
- Not allowed: `sklearn`, `scipy`, and `faiss`. A library seeding routine would not follow this protocol.

## Done looks like
All public tests pass, then **Submit** runs the hidden tests. They check: exact agreement with an independent implementation of the same protocol on several datasets and seeds; that the second centre's frequencies over many seeds follow the mixture of $D^2$ weights, not uniform weights; that $k = 1$ and $k = n$ behave as the protocol says; that the centres are copies of distinct rows; that the output is invariant under an exact power-of-two rescaling of the data; the validation rules, including the distinct-row requirement and the seed rule; that the same seed reproduces the output; and that the inputs are left unchanged.

The hidden tests draw their bulk inputs from the real dataset `california_housing` (see
`docs/DATASETS.md`).

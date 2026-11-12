# Lloyd's k-means from given centroids

Implement Lloyd's algorithm for k-means clustering, starting from the centroids you are given. Write `kmeans(X, init_centroids, *, max_iter=300, tol=1e-6)`, which returns `(centroids, labels, inertia, n_iter)`.

Let $X \in \mathbb{R}^{n \times d}$ hold the samples, and let $C^{(0)} \in \mathbb{R}^{k \times d}$ be `init_centroids`. Repeat for $t = 1, 2, \dots$:

1. **Assign.** Give each sample $\mathbf{x}_i$ the label of its nearest centroid, $\ell_i = \arg\min_j \lVert \mathbf{x}_i - \mathbf{c}_j^{(t-1)} \rVert_2^2$. On an exact tie, take the lower centroid index.
2. **Update.** Set each centroid to the mean of the samples labelled with it, $\mathbf{c}_j^{(t)} = \frac{1}{|S_j|}\sum_{i \in S_j} \mathbf{x}_i$, where $S_j = \{i : \ell_i = j\}$. If $S_j$ is empty, the centroid keeps its previous position, $\mathbf{c}_j^{(t)} = \mathbf{c}_j^{(t-1)}$.
3. **Stop test.** Let $s_t = \max_j \lVert \mathbf{c}_j^{(t)} - \mathbf{c}_j^{(t-1)} \rVert_2$. Stop after this iteration if $s_t \le \texttt{tol}$. Otherwise continue, but never run more than `max_iter` iterations.

Then `n_iter` is the number of iterations run, and the function returns:

- `centroids`: the final centroids, a `float64` array of shape `(k, d)`.
- `labels`: the nearest-centroid label of every sample, computed against the **returned** centroids, as an integer array of shape `(n,)`.
- `inertia`: $\sum_i \lVert \mathbf{x}_i - \mathbf{c}_{\ell_i} \rVert_2^2$, against the returned centroids, as a Python `float`.
- `n_iter`: the number of iterations run, an integer between 1 and `max_iter`.

Reaching `max_iter` without meeting `tol` is not an error. Return the current state.

## Contract
- `X`: 2-D array-like of finite numbers, shape `(n, d)`, with $n \ge 1$ and $d \ge 1$.
- `init_centroids`: 2-D array-like of finite numbers, shape `(k, d)`, with the same `d` as `X` and $1 \le k \le n$.
- `max_iter`: integer, at least 1. Booleans are not integers here.
- `tol`: finite real number, zero or positive.
- Raises: `ValueError` for any violation of the rules above.
- Must not mutate inputs.

## Examples

```python
>>> X = [[0, 0], [0, 1], [10, 10], [10, 11]]
>>> C, labels, inertia, n_iter = kmeans(X, [[0, 0], [10, 10]])
>>> C.tolist(), labels.tolist(), inertia, n_iter
([[0.0, 0.5], [10.0, 10.5]], [0, 0, 1, 1], 1.0, 2)
>>> C, labels, inertia, n_iter = kmeans([[0, 0], [0, 1], [10, 10]], [[0, 0], [100, 100]])
>>> C.tolist()
[[3.3333333333333335, 3.6666666666666665], [100.0, 100.0]]
>>> kmeans(X, [[0, 0, 0]])
ValueError: init_centroids must have the same number of columns as X
```

In the first example, the first update moves both centroids by $0.5$. The second update changes nothing, so the run stops after 2 iterations. In the second example, the second centroid attracts no samples, so it stays at $(100, 100)$.

## Constraints
- Allowed: NumPy and the standard library (`math`). Write the assignment and update steps yourself.
- Not allowed: `sklearn`, `scipy`, and `faiss`. A library clustering routine would skip the steps this problem tests.

## Done looks like
All public tests pass, then **Submit** runs the hidden tests. They check: agreement with scikit-learn's Lloyd implementation on well-separated clusters started from the same centroids; agreement with an independent plain-loop implementation, including the iteration count, the labels and the inertia; the empty-cluster rule; the lower-index tie rule; the stopping rule, including a shift exactly equal to `tol`; `max_iter` as a cap; labels that agree with the returned centroids; inertia that never increases as iterations are added; a large common offset that must not change the clustering; the input validation rules; and that the inputs are left unchanged.

The hidden tests draw their bulk inputs from the real dataset `california_housing` (see
`docs/DATASETS.md`).

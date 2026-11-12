# K-nearest-neighbour regressor

A KNN regressor predicts a number for a query by averaging the targets of its nearest training rows. Implement `knn_regress(X_train, y_train, X_query, k, *, weights="uniform")`, which returns one prediction per query row.

The neighbours of each query are chosen exactly as in a KNN classifier: Euclidean distance $d_j = \lVert \mathbf{x}_j - \mathbf{q} \rVert_2$, ordered by distance with equal distances broken by the lower training index, and the first $k$ rows kept.

- `weights="uniform"`: the prediction is the plain mean of the $k$ neighbours' targets.
- `weights="distance"`: each neighbour has weight $w_j = 1/d_j$, and the prediction is $\sum_j w_j y_j / \sum_j w_j$.
- **Zero distance.** With `weights="distance"`, a neighbour at distance exactly $0$ would have infinite weight. If one or more of the $k$ neighbours sit at distance exactly $0$, the prediction is the plain mean of the targets of those zero-distance neighbours, and the other neighbours are ignored. `weights="uniform"` has no such rule.

## Contract
- `X_train`: 2-D array-like of finite numbers, shape `(n, d)`, with $n \ge 1$ and $d \ge 1$.
- `y_train`: 1-D array-like of finite numbers, shape `(n,)`.
- `X_query`: 2-D array-like of finite numbers, shape `(m, d)`, with the same `d` as `X_train`. $m = 0$ is allowed.
- `k`: integer, with $1 \le k \le n$. Booleans are not integers here.
- `weights`: the string `"uniform"` or `"distance"`. Any other value raises `ValueError`.
- Returns: a 1-D `float64` array of shape `(m,)`.
- Raises: `ValueError` for an invalid `k` or `weights`, for bad shapes or column counts, for non-finite values in `X_train`, `y_train` or `X_query`, or for a `y_train` whose length differs from `X_train`.
- Must not mutate inputs.

## Examples

```python
>>> X, y = [[0.0], [1.0], [2.0], [10.0]], [0.0, 1.0, 2.0, 10.0]
>>> knn_regress(X, y, [[1.2]], k=2)
array([1.5])
>>> knn_regress(X, y, [[1.2]], k=2, weights="distance")
array([1.2])
>>> knn_regress([[0.0], [0.0], [3.0]], [4.0, 8.0, 100.0], [[0.0]], k=3, weights="distance")
array([6.])
>>> knn_regress(X, y, [[1.2]], k=2, weights="median")
ValueError: weights must be "uniform" or "distance"
```

In the second example the neighbours are at distances $0.2$ and $0.8$, so the weights are $5$ and $1.25$, and $(5 \cdot 1 + 1.25 \cdot 2)/6.25 = 1.2$. In the third, two training rows sit exactly on the query, so the zero-distance rule averages $4$ and $8$.

## Constraints
- Allowed: NumPy and the standard library (`math`).
- Not allowed: `sklearn`, `scipy`, and approximate-search libraries (`faiss`, `annoy`, `hnswlib`).

## Done looks like
All public tests pass, then **Submit** runs the hidden tests. They check: agreement with scikit-learn's KNN regressor for both weightings on continuous data; agreement with a brute-force reference that includes exact duplicates; the zero-distance rule, and that it applies only to `"distance"`; $k = 1$ and $k = n$; rejection of an unknown `weights` value, bad `k`, shapes, non-finite values and length mismatches; an empty query; the output being a float64 array with no warnings; a translation of all points by a large constant leaving the predictions unchanged; predictions lying between the smallest and largest neighbour target; and that the inputs are left unchanged.

The hidden tests draw their bulk inputs from the real dataset `california_housing` (see
`docs/DATASETS.md`).

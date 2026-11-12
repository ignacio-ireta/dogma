# K-nearest-neighbour classifier

A k-nearest-neighbour (KNN) classifier stores the training set and classifies a query by looking at its neighbours. Implement `knn_classify(X_train, y_train, X_query, k)`, which returns the predicted class for every query row.

For each query $\mathbf{q}$, compute the Euclidean distance $d_j = \lVert \mathbf{x}_j - \mathbf{q} \rVert_2$ to every training row $j$. Order the training rows by distance, and break equal distances by the lower training index. The first $k$ rows in that order are the neighbours. Count the neighbours' labels and take the most frequent label as the prediction.

**Vote ties.** If several labels share the highest count, the prediction is the tied label of the nearest tied neighbour. Scan the $k$ neighbours from nearest to farthest, and return the label of the first neighbour whose label is among the tied labels. A smaller label value does not win a tie.

## Contract
- `X_train`: 2-D array-like of finite numbers, shape `(n, d)`, with $n \ge 1$ and $d \ge 1$.
- `y_train`: 1-D array-like of integer class labels, shape `(n,)`. Labels may be any integers, including negative ones.
- `X_query`: 2-D array-like of finite numbers, shape `(m, d)`, with the same `d` as `X_train`. $m = 0$ is allowed.
- `k`: integer, with $1 \le k \le n$. Booleans are not integers here.
- Returns: a 1-D integer array of shape `(m,)` containing a label from `y_train` for each query row.
- Raises: `ValueError` when `k` is not an integer or is outside $[1, n]$, when an array is not 2-D or 1-D as required, when the column counts differ, when `y_train` does not have an integer dtype (floats such as `[0.0, 1.0]` are rejected), or when a feature is not finite.
- Features may be as large as `1e200`. Compare distances without overflow, so that unequal finite distances never tie.
- Equal distances. Compute each distance from the coordinate differences $x_j - q$. Two rows whose differences have the same absolute values, coordinate by coordinate, are at exactly the same distance, and the lower index is nearer. Two rows at the same distance on paper but with different differences, such as $(1, 3, 3)$ and $(3, 3, 1)$, may or may not tie after rounding: either order is accepted, and the tests do not depend on it.
- Must not mutate inputs.

## Examples

```python
>>> X = [[0, 0], [1, 0], [0, 1], [5, 5]]
>>> knn_classify(X, [0, 1, 1, 0], [[0.2, 0.2]], k=3)
array([1])
>>> knn_classify([[0, 0], [2, 0], [1, 0], [9, 9]], [7, 3, 5, 9], [[1.2, 0]], k=2)
array([5])
>>> knn_classify([[0, 0], [1, 0]], [4, 2], [[0.5, 0]], k=1)
array([4])
>>> knn_classify(X, [0, 1, 1, 0], [[0.2, 0.2]], k=5)
ValueError: k must be between 1 and the number of training rows
```

The second example is a vote tie: labels 3 and 5 each get one vote among the two neighbours. The neighbour at distance $0.2$ has label 5, so the answer is 5. The third example is an equal-distance tie: both points are at distance $0.5$, and the lower index wins.

## Constraints
- Allowed: NumPy and the standard library (`math`, `collections`). Write the distance and vote yourself.
- Not allowed: `sklearn`, `scipy`, and approximate-search libraries (`faiss`, `annoy`, `hnswlib`). A library neighbour search would skip the thing this problem is about.

## Done looks like
All public tests pass, then **Submit** runs the hidden tests. They check: agreement with a brute-force reference on random data, including many equal distances (rows with the same coordinate differences up to sign) and many vote ties; agreement with scikit-learn's brute-force classifier on tie-free binary data with odd $k$; that equal distances are ordered by training index; that the nearest tied class wins a vote tie, not the smallest label; $k = 1$ and $k = n$; the $k$ bounds and shape, dtype and finiteness rules; an empty query; a translation of all points by a large constant, which must not change any prediction; and that the inputs are left unchanged.

The hidden tests draw their bulk inputs from the real dataset `california_housing` (see
`docs/DATASETS.md`).

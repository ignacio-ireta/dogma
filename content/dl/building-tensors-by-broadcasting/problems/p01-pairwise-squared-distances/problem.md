# Pairwise squared distances

Implement `pairwise_sq_distances(X, Y)`, which returns the matrix of squared Euclidean distances
between every row of `X` and every row of `Y`. Build the result with NumPy broadcasting: no Python
loop over rows or pairs. The result must stay accurate when the points sit far from the origin.

## Contract

- `X`: shape $(n, d)$; `Y`: shape $(m, d)$. Real numbers; integer arrays are allowed. $n$ and $m$
  may be 0. $d \geq 1$. Both are 2-D NumPy arrays (or array-likes that become 2-D arrays).
- Returns: a `float64` array of shape $(n, m)$ with `D[i, j]` equal to $\lVert x_i - y_j \rVert^2$.
- Every entry is $\geq 0$. Rounding must never produce a negative distance.
- Accuracy: when all points share a large offset (for example $10^8$ in every coordinate), the
  distances between them must still match the direct computation to a relative $10^{-6}$.
- Raises `ValueError` when `X` or `Y` is not 2-D, or when the feature counts differ ($d_X \neq d_Y$).
- Must not mutate its inputs.

## Examples

```python
>>> pairwise_sq_distances(np.array([[0., 0.], [1., 0.]]), np.array([[0., 1.]]))
array([[1.],
       [2.]])
>>> X = np.array([[1., 2.], [3., 4.]])
>>> pairwise_sq_distances(X, X)      # zero on the diagonal, 8 off it
array([[0., 8.],
       [8., 0.]])
>>> pairwise_sq_distances(np.zeros((2, 3)), np.zeros((4, 2)))
ValueError: feature counts differ: X has 3 features, Y has 2
```

## Constraints

- Allowed: NumPy, including broadcasting, `np.sum`, `np.asarray`, and `@` or `np.dot` if you use
  the expansion.
- Not allowed: `scipy` and `sklearn` (they are the oracle, not the tool). No Python loop over rows
  or pairs.

## Done looks like

All public tests pass. Then **Submit** runs the hidden tests. They compare against SciPy's
`cdist(..., "sqeuclidean")` on random inputs of several shapes, including square and rectangular
cases, single points, and single features. They check symmetry and a zero diagonal for self-distances,
and non-negativity. They cover the empty-set shapes, feature-count mismatches (including a case
where one side has a single feature, which broadcasting would silently stretch), and non-2-D input.
A precision family checks points far from the origin with small separations. Non-negativity is
also checked on self-distances at large coordinates. A loop check runs the function on 200 points
and fails a solution that runs Python code once per row or per pair. Also covered: integer input
giving a `float64` result, and inputs left unchanged.

The hidden tests draw their bulk inputs from the real datasets `breast_cancer` and `digits` (see
`docs/DATASETS.md`).

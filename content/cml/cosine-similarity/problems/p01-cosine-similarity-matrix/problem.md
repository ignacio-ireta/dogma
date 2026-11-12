# Cosine similarity matrix

Implement `cosine_similarity_matrix(A, B)`, which compares every row of `A` with every row of `B`
and returns the matrix of cosine similarities. Entry `[i, j]` of the result is the cosine of the
angle between row `i` of `A` and row `j` of `B`.

## Contract

- `A`: 2-D array-like of real numbers, shape `(n, d)` with `n >= 1` and `d >= 1`. Integers allowed.
- `B`: 2-D array-like of real numbers, shape `(m, d)` with `m >= 1`. Same `d` as `A`.
- Returns: NumPy array of shape `(n, m)` and dtype `float64`, with values clipped to `[-1, 1]`.
- Zero-norm convention: if row `A[i]` or row `B[j]` is the zero vector, `S[i, j] = 0.0`. This
  matches scikit-learn.
- Raises `ValueError` when `A` or `B` is not 2-D, when their column counts differ, when either
  input is empty (no rows or no columns), or when any entry is NaN or infinite.
- Must not mutate its inputs.

## Examples

```python
>>> cosine_similarity_matrix([[1, 0], [0, 1]], [[1, 1]])
array([[0.70710678],
       [0.70710678]])
>>> cosine_similarity_matrix([[3, 4]], [[6, 8], [-4, 3], [0, 0]])
array([[1., 0., 0.]])     # same direction, orthogonal, zero row
>>> cosine_similarity_matrix([[1, 2]], [[1, 2, 3]])
ValueError: feature count mismatch: 2 vs 3
```

## Constraints

- Allowed: NumPy (array operations, `@`, `np.max`, `np.abs`, `np.where`, `np.clip`, …).
- Not allowed: `sklearn` and `scipy`. Build the normalise-then-multiply pipeline yourself.

## Done looks like

All public tests pass, then **Submit** passes the hidden tests. Those check agreement with an
independent reference on random matrices (including zero rows); symmetry and a unit diagonal for
self-similarity; invariance to positive row scaling; that the result is a transpose when the
arguments are swapped; values staying in `[-1, 1]`; exact values for opposite, parallel and
orthogonal pairs; shape and dtype for integer input; the error cases in the contract; huge and tiny
entries that overflow or underflow a naive norm; and that inputs are left unchanged.

The hidden tests draw their bulk inputs from the real datasets `breast_cancer` and `digits` (see
`docs/DATASETS.md`).

# Gauss-Jordan inverse

Implement `gauss_jordan_inverse(A)`, which returns the inverse of a square matrix. Run
Gauss-Jordan elimination on the augmented matrix `[A | I]` with partial pivoting, until the left
block is the identity. The right block is then `A⁻¹`. A singular or numerically singular matrix
raises `ValueError`.

## Contract

- `A`: 2-D array-like of real numbers, shape `(n, n)` with `n >= 1`. Integers allowed.
- Returns: NumPy array of shape `(n, n)` and dtype `float64`, equal to `A⁻¹`.
- Partial pivoting: in each column, before eliminating, swap in the row at or below the current
  pivot row whose entry has the largest absolute value.
- Singularity: at any step, if the pivot satisfies `|p| <= 1e-12 * max|A|`, where `max|A|` is the
  largest absolute entry of the original `A`, raise `ValueError("matrix is singular or numerically singular")`.
  The zero matrix is singular.
- Raises `ValueError` when `A` is not 2-D, when it is not square, when it is empty, when it contains
  NaN or infinity, or when it is singular as defined above.
- Must not mutate its input.

## Examples

```python
>>> gauss_jordan_inverse([[4, 7], [2, 6]])
array([[ 0.6, -0.7],
       [-0.2,  0.4]])
>>> gauss_jordan_inverse([[0, 1], [1, 0]])     # first pivot is zero: swap rows first
array([[0., 1.],
       [1., 0.]])
>>> gauss_jordan_inverse([[1, 2], [2, 4]])
ValueError: matrix is singular or numerically singular
>>> gauss_jordan_inverse([[1, 2, 3], [4, 5, 6]])
ValueError: expected a square matrix, got shape (2, 3)
```

## Constraints

- Allowed: NumPy array operations (`np.array`, `np.eye`, `np.hstack`, `np.argmax`, `np.abs`, `np.outer`, …).
- Not allowed: `numpy.linalg` and `scipy`. Compute the inverse yourself.

## Done looks like

All public tests pass, then **Submit** passes the hidden tests. Those check agreement with an
independent inverse on random well-conditioned matrices; `A @ inverse ≈ I` and inverting twice
returns `A`; the transpose and scaling identities; matrices whose first pivot is zero; permutation
matrices; a badly scaled but nonsingular matrix (the threshold is relative, so it must not be
rejected); singular and nearly singular input raising `ValueError`, with the threshold checked on
both sides (a pivot far above it must still be accepted); non-square, 1-D, empty and non-finite
input raising `ValueError`; integer input returning `float64`; and inputs left unchanged.

The hidden tests draw their bulk inputs from the real datasets `breast_cancer` and `wine` (see
`docs/DATASETS.md`).

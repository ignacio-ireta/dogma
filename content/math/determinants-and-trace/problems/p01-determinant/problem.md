# Determinant

Implement `determinant(A)`, which returns the determinant of a square matrix. Use LU-style
elimination with partial pivoting, and track the sign flipped by each row swap. The determinant is
the sign times the product of the pivots. Do not expand cofactors.

## Contract

- `A`: 2-D array-like of real numbers, shape `(n, n)` with `n >= 0`. Integers allowed. The empty
  `0 x 0` matrix is valid.
- Returns: the determinant as a float. For the `0 x 0` matrix, return `1.0`.
- Algorithm: partial pivoting. Each row swap multiplies the sign by `-1`. If some column has no
  nonzero entry on or below the diagonal, return `0.0`.
- Raises `ValueError` when `A` is not 2-D, when it is not square, or when it contains NaN or infinity.
- Must not mutate its input.
- A large matrix can overflow or underflow the product of pivots. This function does not have to
  guard against that.

## Examples

```python
>>> determinant([[1, 2], [3, 4]])     # the largest entry in column 1 is 3, so rows swap
-2.0
>>> determinant([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
1.0
>>> determinant([[1, 2], [2, 4]])     # second row is twice the first
0.0
>>> determinant([[1, 2, 3], [4, 5, 6]])
ValueError: expected a square matrix, got shape (2, 3)
```

## Constraints

- Allowed: NumPy array operations (`np.array`, `np.argmax`, `np.abs`, `np.outer`, `np.isfinite`, …).
- Not allowed: `numpy.linalg` and `scipy`. Cofactor expansion is correct but does not finish at
  `n = 30`, so the hidden tests will time out on it.

## Done looks like

All public tests pass, then **Submit** passes the hidden tests. Those check agreement with an
independent determinant on random well-conditioned matrices, up to `n = 30`; `det(AB) = det(A) det(B)`;
`det(Aᵀ) = det(A)`; triangular matrices (upper and lower) giving the product of the diagonal; the sign
flip under a row swap; the scaling of a row and of the whole matrix; the invariance under adding a
multiple of one row to another; all 24 permutation matrices of size 4 giving their sign; singular
matrices giving zero; the `0 x 0` and `1 x 1` cases; invalid input raising `ValueError`; integer
input; and inputs left unchanged.

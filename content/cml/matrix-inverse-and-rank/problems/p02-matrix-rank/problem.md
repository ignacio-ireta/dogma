# Matrix rank

Implement `matrix_rank(A, tol=None)`, which returns the rank of a matrix by row reduction. Walk
through the columns with Gaussian elimination and partial pivoting. Each column that yields a usable
pivot adds one to the rank. A column whose remaining entries are all negligible adds nothing, because
it depends on the columns before it.

## Contract

- `A`: 2-D array-like of real numbers, shape `(m, n)` with `m >= 1` and `n >= 1`. Integers allowed.
- `tol`: optional non-negative float. If omitted, use
  `tol = 1000 * max(m, n) * eps * ||A||_F`, where `eps` is float64 machine epsilon and `||A||_F` is
  the Frobenius norm, the square root of the sum of squared entries.
- Algorithm: for each column, left to right, find the largest-magnitude entry among the rows that are
  not yet pivot rows. If its magnitude is `<= tol`, skip the column. Otherwise swap that row into the
  next pivot position and eliminate the entries below it. The rank is the number of pivot rows.
- Returns: the rank as an `int`, with `0 <= rank <= min(m, n)`.
- Raises `ValueError` when `A` is not 2-D, when it is empty, when it contains NaN or infinity, or when
  `tol` is negative.
- Must not mutate its input.

## Examples

```python
>>> matrix_rank([[1, 2], [3, 4]])
2
>>> matrix_rank([[1, 2, 3], [4, 5, 6], [7, 8, 9]])   # row 3 = 2 * row 2 - row 1
2
>>> matrix_rank([[1, 2], [2, 4]])                    # second row is twice the first
1
>>> matrix_rank([[1, 0], [0, 1e-10]])                # default tolerance keeps the small pivot
2
>>> matrix_rank([[1, 0], [0, 1e-10]], tol=1e-8)      # an explicit tolerance drops it
1
```

## Constraints

- Allowed: NumPy array operations (`np.array`, `np.argmax`, `np.abs`, `np.outer`, `np.finfo`, …).
- Not allowed: `numpy.linalg` (no `matrix_rank`, `norm` or SVD) and `scipy`.

## Done looks like

All public tests pass, then **Submit** passes the hidden tests. Those check agreement with NumPy's
rank on constructed matrices of known rank (wide, tall, square, rank zero, rank one), including
full-rank rectangular matrices; invariance of the rank under transposition, row permutation, and
scaling a row by a nonzero factor; that adding a row that combines existing rows leaves the rank
unchanged; rank one for an outer product; identity and zero matrices; the effect of an explicit
tolerance on a tiny pivot, including one equal to it; the default tolerance on nearly collinear pairs
(square and tall) and on a clearly nonzero small pivot; invalid input and negative `tol` raising `ValueError`; integer input; and inputs left
unchanged.

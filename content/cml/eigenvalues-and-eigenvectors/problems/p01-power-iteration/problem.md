# Power iteration

Implement `power_iteration(A, num_iters=1000, tol=1e-10)`, which finds the dominant eigenpair of a
real symmetric matrix: the eigenvalue of largest absolute value, and a unit eigenvector for it. Use
repeated multiplication and normalisation, not `numpy.linalg`.

## Contract

- `A`: 2-D array-like of real numbers, shape `(n, n)` with `n >= 1`, and symmetric. Integers allowed.
- Symmetry: raise `ValueError` when `max|A - Aᵀ| > 1e-9 * max(1, max|A|)`.
- `num_iters`: integer `>= 1`. `tol`: float `> 0`. Raise `ValueError` for other values.
- Start vector: a fixed, deterministic vector (no unseeded randomness), the same on every call.
  It must be generic. A constant vector such as `np.ones(n)` can be exactly orthogonal to the
  dominant eigenvector of a symmetric matrix; the iteration then converges to a different
  eigenpair without raising any error.
- Stopping: after each update, compute the Rayleigh quotient `λ = vᵀAv` for the unit vector `v`, and
  the residual `r = Av - λv`. Stop when `‖r‖ <= tol * max(1, |λ|)`, or after `num_iters` updates. Using
  up the budget is not an error: return the current estimate.
- Returns: `(eigenvalue, eigenvector)`. `eigenvalue` is a Python float: the eigenvalue of largest
  absolute value, with its sign. `eigenvector` is a float64 array of shape `(n,)` with unit length. Sign
  convention: the first entry whose absolute value is within a relative `1e-9` of the largest absolute
  entry is positive.
- The zero matrix returns `(0.0, unit vector)`.
- Raises `ValueError` when `A` is not square, contains NaN or infinity, is not symmetric, or when
  `num_iters` or `tol` is out of range.
- Must not mutate its input.

## Examples

```python
>>> value, vector = power_iteration([[2, 1], [1, 2]])
>>> round(value, 6), np.round(vector, 6)
(3.0, array([0.707107, 0.707107]))
>>> power_iteration(np.diag([-5.0, 4.0]))[0]     # largest |eigenvalue| is -5, and it keeps its sign
-5.0
>>> power_iteration([[1, 2], [0, 1]])
ValueError: matrix must be symmetric
```

## Constraints

- Allowed: NumPy array operations (`@`, `np.sqrt`, `np.abs`, `np.argmax`, `np.random.default_rng`, …).
- Not allowed: `numpy.linalg` (no `eig`, `eigh`, `norm` or `solve`) and `scipy`.

## Done looks like

All public tests pass, then **Submit** passes the hidden tests. Those check agreement with a symmetric
eigensolver on random symmetric matrices with a clear gap, including matrices whose dominant eigenvalue
is negative; that the output is an eigenpair with unit length; the sign convention; that the dominant
eigenvalue is chosen by magnitude rather than by value; a matrix on which a start vector with no
component along the dominant eigenvector converges to the wrong pair; a slow but convergent gap within
the default budget; the zero and `1 x 1` matrices; invalid input and parameters; integer input; and
inputs left unchanged.

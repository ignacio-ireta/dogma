# Normal-equation linear regression

Implement `fit_linear_regression(X, y, fit_intercept=True)`. It fits `y ≈ intercept + X @ coef` by
least squares, minimising ‖y − A β‖² where the design matrix `A` is `[1 | X]` when `fit_intercept` is
true and `X` otherwise. Solve the normal equations `(Aᵀ A) β = Aᵀ y` with NumPy. If the design matrix
is rank deficient, the coefficients are not unique, so raise an error.

## Contract

- `X`: 2-D array-like, shape (n, d), d ≥ 1, finite real numbers. Integers allowed.
- `y`: 1-D array-like, shape (n,), finite real numbers.
- `fit_intercept`: bool, default `True`.
- Returns: `(intercept, coef)`. `intercept` is a Python float, and it is `0.0` when `fit_intercept`
  is false. `coef` is a float64 array of shape (d,).
- Raises `ValueError` when `X` is not 2-D; when `y` is not 1-D or has a different length from `X`;
  when `X` or `y` contains NaN or infinity; when the design matrix is rank deficient. The rank is the
  one NumPy computes with `np.linalg.matrix_rank` and its default tolerance. This covers a column that
  is a multiple or a sum of others, a zero column, a constant column when `fit_intercept` is true, and
  fewer rows than columns.
- Must not mutate `X` or `y`.

## Examples

```python
>>> fit_linear_regression([[1.], [2.], [3.]], [3., 5., 7.])
(1.0, array([2.]))                 # y = 1 + 2x exactly (up to rounding)
>>> fit_linear_regression([[1.], [2.], [3.]], [2., 4., 6.], fit_intercept=False)
(0.0, array([2.]))                 # a line through the origin
>>> fit_linear_regression([[1., 3.], [2., 6.], [3., 9.]], [1., 2., 3.])
ValueError                         # the second column is three times the first
```

## Constraints

- Allowed: NumPy arrays, `np.linalg.solve` and `np.linalg.matrix_rank`.
- Not allowed: `sklearn`, `scipy`, and the solvers `numpy.linalg.lstsq`, `numpy.linalg.pinv`,
  `numpy.linalg.inv` and `numpy.linalg.qr`. Solve the normal equations. Do not invert or factor the
  design matrix.

## Done looks like

All public tests pass, then **Submit** passes the hidden tests. Those check agreement with
scikit-learn's `LinearRegression` and with NumPy's least-squares solver on random data, with and
without an intercept; the normal equations themselves (residuals orthogonal to every design column,
including the intercept column); a finite-difference check that the squared-error gradient vanishes
at the returned fit; exact recovery of a noise-free line; a nearly collinear full-rank design whose
coefficients must match the fit; rank-deficient designs of several kinds, including scaled duplicate
columns and column sums that a plain solve accepts silently, all raising `ValueError`; too few rows;
invalid shapes and non-finite values; equivariance when `y` or a feature is rescaled, and when a
feature is shifted; very large and very small feature scales; integer input; and inputs left
unchanged.

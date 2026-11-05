# Ridge regression

Implement `ridge_fit(X, y, alpha, *, fit_intercept=True)`. Ridge regression adds an L2 penalty to
least squares. It minimises ‖y − b₀·1 − Xβ‖² + α‖β‖² over the intercept b₀ and the coefficients β.
The intercept is **not** penalised. Solve the centred normal equations directly with NumPy. Do not use
a least-squares or pseudo-inverse routine. With α > 0 the system is solvable even when `X` has linearly
dependent columns.

## Contract

- `X`: 2-D array-like of shape (n, d), n ≥ 1, d ≥ 1, finite real numbers. Integers allowed.
- `y`: 1-D array-like of shape (n,), finite real numbers.
- `alpha`: a finite real number ≥ 0, the penalty strength.
- `fit_intercept`: bool, default `True`. When `False`, the intercept is fixed at 0 and the problem is
  ‖y − Xβ‖² + α‖β‖².
- Returns: `(intercept, coef)`. `intercept` is a Python float, and it is `0.0` when `fit_intercept` is
  false. `coef` is a float64 array of shape (d,).
- With `fit_intercept` true, centre the data: X̃ = X − x̄ and ỹ = y − ȳ. Then
  β = (X̃ᵀX̃ + αI)⁻¹ X̃ᵀỹ and b₀ = ȳ − x̄ᵀβ. With `fit_intercept` false, β = (XᵀX + αI)⁻¹ Xᵀy.
- `alpha = 0` gives ordinary least squares. That needs a full-rank design: the centred X̃ when
  `fit_intercept` is true, and X otherwise. Raise `ValueError` if it is rank deficient.
- Raises `ValueError` when `X` has no rows or no columns; when `X` is not 2-D; when `y` is not 1-D or its length differs from the number of
  rows of `X`; when `X`, `y` or `alpha` contains a non-finite value; when `alpha < 0`; or when
  `alpha = 0` with a rank-deficient design. Use `np.linalg.matrix_rank` with its default tolerance.
- Must not mutate inputs.

## Examples

```python
>>> ridge_fit([[1.], [2.], [3.]], [1., 2., 2.], 2.0)
(1.1666666666666667, array([0.25]))          # slope 1/(2+2); intercept 5/3 - 2 * 0.25
>>> ridge_fit([[1.], [2.], [3.]], [1., 2., 2.], 0.0)
(0.6666666666666667, array([0.5]))           # ordinary least squares
>>> ridge_fit([[1., 3.], [2., 6.], [3., 9.]], [1., 2., 3.], 0.0)
ValueError   # the second column is three times the first, so alpha = 0 has no unique solution
```

## Constraints

- Allowed: NumPy, including `np.linalg.solve` and `np.linalg.matrix_rank`.
- Not allowed: `sklearn`, `scipy`, `numpy.linalg.lstsq` and `numpy.linalg.pinv`. Solve the normal
  equations; do not call a least-squares or pseudo-inverse routine.

## Done looks like

All public tests pass, then **Submit** passes the hidden tests. Those compare the fit with scikit-learn's
`Ridge` (with and without an intercept, and on a rank-deficient design with α > 0) and with the SVD form
of the same solution. They check that α → 0 recovers least squares, that the coefficient norm shrinks as α
grows, and that large α sends the coefficients to 0 and the intercept to ȳ. They check that the intercept
is unpenalised (shifting y moves only the intercept, and shifting X moves only the intercept), that
rescaling a feature with α scaled by the square of the same factor rescales its coefficient, and that α = 0
on a rank-deficient design is rejected. They also cover invalid inputs, integer input, and unchanged inputs.

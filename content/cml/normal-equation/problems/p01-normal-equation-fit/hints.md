## Hint 1 — concept

Least squares chooses the coefficients that make the residual orthogonal to every column of the
design, so the fitted values are the projection of `y` onto the column space. Setting the gradient
of the squared error to zero gives the normal equations. A unique solution exists only when the
design has full column rank.

## Hint 2 — formula / approach

Build `A = [1 | X]` when `fit_intercept` is true, or `A = X` otherwise. Check `np.linalg.matrix_rank(A)`
against the number of columns of `A`, and raise `ValueError` if it is smaller. Then solve
`(AᵀA) β = Aᵀy` with `np.linalg.solve`. Check the rank of `A` itself, not of `AᵀA`, because forming
`AᵀA` squares the condition number first.

## Hint 3 — code shape

`A = np.hstack([np.ones((n, 1)), X_arr])` when fitting an intercept. Then
`beta = np.linalg.solve(A.T @ A, A.T @ y_arr)`. Return `(float(beta[0]), beta[1:].copy())` when the
intercept is fitted, and `(0.0, beta)` otherwise. Validate `X` (2-D), `y` (1-D with length `n`) and
finiteness before building `A`.

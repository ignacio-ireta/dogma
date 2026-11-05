## Hint 1 — concept

Ridge adds α times the squared coefficient norm to the least-squares objective. Centre X and y first.
That takes the intercept out of the penalty, so you solve for the slopes alone and recover the
intercept afterwards from the means.

## Hint 2 — formula / approach

With the centred matrix Xc and the centred target yc, the slopes solve (Xcᵀ Xc + α I) β = Xcᵀ yc. Then the
intercept is ȳ − x̄ᵀβ. With α > 0 the matrix is positive definite, so a dependent design is still solvable.
With α = 0 the design must have full column rank, so check `np.linalg.matrix_rank` on the centred X first.

## Hint 3 — code shape

Compute `x_mean = X.mean(axis=0)` and `xc = X - x_mean`, then the Gram matrix `xc.T @ xc + alpha * I`.
Solve it with `np.linalg.solve` against `xc.T @ yc`. Return `(float(y_mean - x_mean @ beta), beta)`, or
`(0.0, beta)` without an intercept. Keep the validation block at the top of the function.

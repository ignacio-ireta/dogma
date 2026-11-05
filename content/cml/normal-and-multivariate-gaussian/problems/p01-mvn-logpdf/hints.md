## Hint 1: concept

The log-density has three parts: a constant, a log-determinant term that measures the volume of the
covariance ellipsoid, and a squared Mahalanobis distance. Each part is a sum or a quadratic form, so
none needs the matrix inverse or the determinant themselves.

## Hint 2: formula / approach

Factor $\Sigma = LL^\top$ with `np.linalg.cholesky`. Then $\log\det\Sigma = 2\sum_i \log L_{ii}$, and
$(x-\mu)^\top\Sigma^{-1}(x-\mu) = \lVert z\rVert^2$ where $Lz = x-\mu$. Check symmetry and positive
definiteness before factoring: `np.linalg.cholesky` raises `LinAlgError` for a matrix that is not
positive definite, and it silently reads only one triangle when the matrix is asymmetric.

## Hint 3: code shape

Convert the three inputs with `np.asarray(..., dtype=np.float64)`, check the shapes, then
`chol = np.linalg.cholesky(cov)` inside a `try` that re-raises `LinAlgError` as `ValueError`. Compute
`log_det = 2 * np.sum(np.log(np.diag(chol)))` and `z = np.linalg.solve(chol, (X - mean).T)`. Return
`-0.5*d*np.log(2*np.pi) - 0.5*log_det - 0.5*np.sum(z*z, axis=0)`.

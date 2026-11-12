# Multivariate normal log-density

Implement `mvn_logpdf(X, mean, cov)`, which returns the log-density of the multivariate normal
distribution $\mathcal{N}(\mu, \Sigma)$ at each row of `X`. Work in log space, and never form
$\Sigma^{-1}$ or $\det \Sigma$ explicitly: at dimension 60 the determinant of a perfectly reasonable
covariance underflows to $0$ or overflows to $\infty$, while its logarithm stays finite.

## Contract

- `X`: 2-D array-like of real numbers, shape `(n, d)` with $n \ge 1$ and $d \ge 1$. Every entry is finite.
  Integers allowed.
- `mean`: 1-D array-like of length `d`, finite.
- `cov`: `(d, d)` array-like. It must be symmetric and positive definite.
- Returns: NumPy array of shape `(n,)` and dtype `float64`. Element $i$ is
  $$\log \mathcal{N}(x_i \mid \mu, \Sigma) = -\tfrac{d}{2}\log 2\pi - \tfrac12 \log\det\Sigma
  - \tfrac12 (x_i-\mu)^\top \Sigma^{-1} (x_i-\mu).$$
- Raises `ValueError` when `X` is not 2-D, when it has no rows or no columns, when `mean` does not have
  length `d`, when `cov` is not `(d, d)`, when any input contains NaN or infinity, when `cov` is not
  symmetric (entries that differ by more than $10^{-10}\max(1, \max|\Sigma|)$), or when `cov` is not
  positive definite (a singular matrix counts as not positive definite).
- Must not mutate its inputs.

## Examples

```python
>>> mvn_logpdf([[0.0]], [0.0], [[1.0]])                       # standard normal at 0
array([-0.91893853])
>>> mvn_logpdf([[0.0, 0.0]], [0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]])
array([-1.83787707])
>>> mvn_logpdf([[1.0]], [0.0], [[4.0]])                       # sigma = 2, x = 1
array([-1.73708571])
>>> mvn_logpdf([[0.0, 0.0]], [0.0, 0.0], [[1.0, 0.5], [0.0, 1.0]])
ValueError: covariance must be symmetric
```

## Constraints

- Allowed: NumPy array operations and linear algebra that is not an explicit inverse or determinant:
  `np.linalg.cholesky`, `np.linalg.solve`, `np.linalg.slogdet`, `np.linalg.eigvalsh`, `np.log`, `np.sum`, ...
- Not allowed: `scipy`, `numpy.linalg.inv` and `numpy.linalg.det`. Use a factorisation and a solve.

## Done looks like

All public tests pass, then **Submit** passes the hidden tests. Those check agreement with
`scipy.stats.multivariate_normal` on random inputs with dimension from 1 to 5; a one-dimensional
closed form; a dimension-60 covariance whose determinant underflows, and another whose determinant
overflows, where the log-density must still be correct; the change-of-variables identity under an
invertible linear map; the error cases in the contract (shape, symmetry, positive definiteness,
non-finite input, and zero observations); a covariance that is symmetric only up to rounding, which must be
accepted; a single observation keeping its shape; integer input; and inputs left unchanged.

The hidden tests draw their bulk inputs from the real datasets `breast_cancer` and `wine` (see
`docs/DATASETS.md`).

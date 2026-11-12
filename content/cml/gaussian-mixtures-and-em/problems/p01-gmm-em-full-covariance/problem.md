# Gaussian mixture EM with full covariances

Implement `gmm_em(X, k, init_means, n_iter, reg)`, which fits a $k$-component Gaussian mixture to
the samples in `X` by expectation-maximisation (EM). Each component has its own full covariance
matrix, and every covariance is kept above a floor `reg`. Return the final weights, means and
covariances, and the log-likelihood after every iteration.

Notation: $X\in\mathbb{R}^{n\times d}$ has rows $\mathbf{x}_i$. Component $j$ has weight $\pi_j$, mean
$\boldsymbol\mu_j$ and covariance $\Sigma_j$. Let $\mathcal{N}_{ij} = \mathcal{N}(\mathbf{x}_i \mid \boldsymbol\mu_j, \Sigma_j)$.

**Floor.** $\operatorname{floor}(S)$ takes the eigendecomposition $S = V\Lambda V^\top$ and returns
$V\max(\Lambda, \mathrm{reg})V^\top$. Each eigenvalue is raised to at least `reg`, and the eigenvectors are kept.

**Initial state.** $\pi_j = 1/k$, $\boldsymbol\mu_j$ is row $j$ of `init_means`, and every
$\Sigma_j = \operatorname{floor}(S_0)$, where $S_0 = \frac1n\sum_i(\mathbf{x}_i - \bar{\mathbf{x}})(\mathbf{x}_i - \bar{\mathbf{x}})^\top$.

**Each iteration**, in order:

1. **E-step.** $r_{ij} = \pi_j\mathcal{N}_{ij} / \sum_l \pi_l\mathcal{N}_{il}$. Compute this in log space.
2. **M-step.** Let $N_j = \sum_i r_{ij}$.
   - If $N_j = 0$, the component is dead: set $\pi_j = 0$ and keep $\boldsymbol\mu_j$ and $\Sigma_j$.
   - Otherwise $\pi_j = N_j / n$, $\boldsymbol\mu_j = \frac{1}{N_j}\sum_i r_{ij}\mathbf{x}_i$, and
     $\Sigma_j = \operatorname{floor}\!\big(\frac{1}{N_j}\sum_i r_{ij}(\mathbf{x}_i - \boldsymbol\mu_j)(\mathbf{x}_i - \boldsymbol\mu_j)^\top\big)$,
     using the **new** mean $\boldsymbol\mu_j$.
3. **Log-likelihood** of the new parameters: $\ell = \sum_i \log\sum_j \pi_j\mathcal{N}_{ij}$.

## Contract
- `X`: 2-D array-like of finite numbers, shape `(n, d)`, with $n \ge 1$ and $d \ge 1$.
- `k`: integer with $1 \le k \le n$. Booleans are not integers here.
- `init_means`: finite array-like of shape `(k, d)`.
- `n_iter`: integer $\ge 0$. Booleans are not integers here.
- `reg`: finite real number $> 0$.
- Returns: a tuple `(weights, means, covs, lls)` of float64 arrays with shapes `(k,)`, `(k, d)`,
  `(k, d, d)` and `(n_iter + 1,)`. `lls[t]` is the log-likelihood after $t$ iterations, so `lls[0]` belongs to the initial state.
- Raises `ValueError` for any violation of the rules above.
- Must not mutate inputs.

## Examples

```python
>>> w, mu, cov, lls = gmm_em([[0.0], [2.0]], 2, [[0.0], [2.0]], n_iter=0, reg=1e-3)
>>> w, lls                      # equal weights; log-likelihood of the initial state
(array([0.5, 0.5]), array([-2.9703]))

>>> w, mu, cov, lls = gmm_em([[0.0], [2.0]], 2, [[0.0], [2.0]], n_iter=1, reg=1e-3)
>>> mu.ravel(), cov.ravel()     # each mean moves toward the sample it is responsible for
(array([0.2384, 1.7616]), array([0.4200, 0.4200]))

>>> gmm_em([[0.0], [2.0]], 3, [[0.0], [2.0], [5.0]], n_iter=1, reg=1e-3)
ValueError: k must be at least 1 and at most the number of samples
```

## Constraints
- Allowed: NumPy, including `np.linalg` (for example `cholesky`, `solve`, `eigh`) and `math`.
  Write the E-step and M-step yourself.
- Not allowed: `sklearn`, whose mixture models run EM for you and skip the steps this problem tests.

## Done looks like
All public tests pass, then **Submit** runs the hidden tests. They check: the one-iteration result
on a two-point fixture, computed by hand from the responsibilities; the initial log-likelihood in
closed form; agreement with an independent reference on a seeded two-dimensional dataset; that
the log-likelihood never decreases across iterations; the covariance floor on duplicate points;
responsibilities for far-apart points, where a naive exponential underflows; a component that
receives no responsibility; invariance to sample order and to a large common offset; shapes and
dtypes; invalid inputs; and that the inputs are left unchanged.

The hidden tests draw their bulk inputs from the real datasets `iris` and `wine` (see
`docs/DATASETS.md`).

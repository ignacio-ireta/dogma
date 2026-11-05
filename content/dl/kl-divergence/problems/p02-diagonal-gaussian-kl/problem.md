# KL divergence between two diagonal Gaussians

Implement `diagonal_gaussian_kl(mu_p, sigma_p, mu_q, sigma_q)`. It returns the
Kullback-Leibler divergence in nats from one diagonal Gaussian to another:

$$P=\mathcal N(\mu_p,\operatorname{diag}(\sigma_p^2)),\qquad Q=\mathcal N(\mu_q,\operatorname{diag}(\sigma_q^2)),$$

where $\mu$ is the mean and $\sigma$ the standard deviation, both vectors of length $d$. This is the
term a variational autoencoder adds to its loss when the encoder outputs a diagonal Gaussian and the
prior is a standard normal. Use a closed form. Do not sample.

## Contract

- `mu_p`, `mu_q`, `sigma_p`, `sigma_q`: one-dimensional real array-likes of the same length
  `d >= 1`. Entries are finite. Integer inputs are accepted.
- Every entry of `sigma_p` and `sigma_q` is strictly positive. Zero and negative standard deviations are invalid.
- The formula must be accurate for standard deviations in $[10^{-50}, 10^{50}]$ and means with
  absolute value at most $10^{50}$. Results can be large, but they must not be computed
  through an overflowing intermediate.
- Returns: a Python `float`,
  $$\mathrm{KL}=\sum_{i=1}^{d}\Big[\log\frac{\sigma_{q,i}}{\sigma_{p,i}}+\frac{\sigma_{p,i}^2+(\mu_{p,i}-\mu_{q,i})^2}{2\,\sigma_{q,i}^2}-\frac12\Big].$$
- Raises `ValueError` when: an input is not 1-D or is empty; the four lengths differ; an input has
  a non-real dtype (bool, complex, object); an entry is `NaN` or infinite; or a standard deviation is not strictly positive.
- Must not mutate its inputs.

## Examples

```python
>>> diagonal_gaussian_kl([0.0], [1.0], [0.0], [1.0])
0.0                           # the same distribution
>>> diagonal_gaussian_kl([0.0], [1.0], [1.0], [1.0])
0.5                           # unit variance, mean shift 1: (0 - 1)^2 / 2
>>> diagonal_gaussian_kl([0.0], [1.0], [0.0], [2.0])
0.3181471805599453            # log 2 + 1/8 - 1/2
```

```python
>>> diagonal_gaussian_kl([0.0, 0.0], [1.0, 1.0], [1.0, 0.0], [1.0, 2.0])
0.8181471805599453            # the two coordinates add: 0.5 + 0.3181471805599453
```

## Constraints

- Allowed: NumPy and the standard library (`math`).
- Not allowed: `scipy.stats` and `scipy.special`. They are not needed for a closed form, and
  the tests use a Monte Carlo estimate as an independent oracle. Write the formula yourself.

## Done looks like

All public tests pass. Then **Submit** runs the hidden tests. They check: the analytic cases
(identity, a mean shift, a scale change, and two coordinates); agreement with a seeded Monte
Carlo estimate on a three-dimensional pair with asymmetric parameters, within a stated number
of standard errors; that the forward and reverse divergences are not confused; that variances are
used rather than standard deviations; the sign of the log-scale term; invariance when all means and
standard deviations are scaled by a common factor across forty orders of magnitude; exact zero for
identical distributions; every `ValueError` rule; a Python `float` return; and inputs left unchanged.

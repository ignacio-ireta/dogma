# Gaussian maximum likelihood

Implement `gaussian_mle(X)`, which returns the maximum-likelihood estimates of the mean and covariance of a
multivariate normal distribution from observations. Each row of `X` is one observation and each column is one
variable. The covariance estimate divides by $n$, not $n-1$, because that is what maximising the likelihood
gives. It is slightly biased, and the lesson explains why that is acceptable here.

## Contract

- `X`: 2-D array-like of real numbers, shape `(n, d)` with $n \ge 1$ rows and $d \ge 1$ columns. Integers
  allowed. Every entry is finite.
- Returns: a tuple `(mean, covariance)` of NumPy arrays with dtype `float64`:
  - `mean`, shape `(d,)`: the column means $\hat\mu = \frac{1}{n}\sum_i x_i$.
  - `covariance`, shape `(d, d)`: $\hat\Sigma = \frac{1}{n}\sum_i (x_i-\hat\mu)(x_i-\hat\mu)^\top$.
  With one observation, `covariance` is the $d \times d$ zero matrix.
- Raises `ValueError` when `X` is not 2-D, when it has no rows or no columns, or when it contains NaN or
  infinity.
- Must not mutate its input.

## Examples

```python
>>> mean, cov = gaussian_mle([[1, 2], [3, 4], [5, 9]])
>>> mean
array([3., 5.])
>>> cov                                    # divided by n = 3
array([[2.6667, 4.6667],
       [4.6667, 8.6667]])
>>> gaussian_mle([[1.0, 2.0]])[1]          # one observation: no spread
array([[0., 0.],
       [0., 0.]])
>>> gaussian_mle(np.empty((0, 2)))
ValueError: need at least one observation and one variable
```

## Constraints

- Allowed: NumPy array operations (`np.asarray`, `.mean`, `.T`, `@`, `np.isfinite`, `np.einsum`, ...).
- Not allowed: `numpy.cov`, `pandas` and `scipy`. Compute the estimates from their definitions.

## Done looks like

All public tests pass, then **Submit** passes the hidden tests. Those check agreement with the biased
NumPy estimates on random data; the relation between the MLE and the unbiased covariance,
$\hat\Sigma = \frac{n-1}{n}\,S$; that the returned parameters score a higher log-likelihood than small
perturbations of them; a single observation; a large common offset of about $10^9$, where the shortcut
$\mathbb{E}[xy]-\mathbb{E}[x]\,\mathbb{E}[y]$ loses all precision; the one-variable shape; NaN, non-2-D and
empty input raising `ValueError`; integer input; and inputs left unchanged.

The hidden tests draw their bulk inputs from the real datasets `breast_cancer` and `wine` (see
`docs/DATASETS.md`).

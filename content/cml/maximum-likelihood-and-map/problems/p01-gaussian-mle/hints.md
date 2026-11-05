## Hint 1: concept

Maximising the Gaussian likelihood over the mean gives the sample mean. Maximising it over the covariance gives
the average outer product of the deviations from that mean. Both are averages, and the covariance average uses
the same count, $n$, as the mean.

## Hint 2: formula / approach

$\hat\mu = \frac{1}{n}\sum_i x_i$ and $\hat\Sigma = \frac{1}{n}\sum_i (x_i-\hat\mu)(x_i-\hat\mu)^\top$. Validate
first: 2-D, at least one row and one column, finite entries. Then the estimates are two lines of NumPy.

## Hint 3: code shape

Convert with `np.asarray(X, dtype=np.float64)`, check `.ndim`, `.shape` and `np.isfinite`. Compute
`mean = x.mean(axis=0)`, then `centred = x - mean` (a new array), and return `mean` with
`centred.T @ centred / n` as a tuple.

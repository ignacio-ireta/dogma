# Poisson probability mass function

Implement `poisson_pmf(k, lam)`. It returns $P(X=k)$ for each entry of `k`, where
$X \sim \text{Poisson}(\lambda)$ counts events in a fixed interval when events arrive at a constant
rate $\lambda$. Computed directly, $\lambda^k$ and $k!$ overflow float64 long before the answer
does: $171!$ is the first factorial beyond the float64 range, and $\lambda^k$ overflows for large $k$. Work in log
space with the log-gamma function from the standard library.

## Contract

- `k`: integer array-like of any shape. Integral floats such as `2.0` are accepted. Values below 0
  have probability `0.0`.
- `lam`: finite real number $\lambda \ge 0$. Mean and variance are both $\lambda$.
- Returns: NumPy `float64` array with the same shape as `k`, holding $P(X=k)$. A scalar `k` gives a
  0-d array.
- Raises `ValueError` when `lam` is negative, NaN or infinite, or when `k` contains a non-integer.
- Must not mutate its inputs.

## Examples

```python
>>> poisson_pmf([0, 1, 2], 2.0)            # e^-2 * 2^k / k!
array([0.13533528, 0.27067057, 0.27067057])
>>> poisson_pmf(-1, 2.0)                   # negative counts are impossible
array(0.)
>>> poisson_pmf(0, 0.0)                    # rate 0: no events, all mass on k = 0
array(1.)
>>> poisson_pmf(200, 180.0)                # k! and 180^200 both overflow float64 when computed directly
array(0.00965174)
>>> poisson_pmf(2.5, 2.0)
ValueError: k must contain integers
```

## Constraints

- Allowed: the standard library `math` (`math.lgamma`, `math.log`, …) and NumPy.
- Not allowed: `scipy`. `scipy.stats.poisson` would give the answer without the work.

## Done looks like

All public tests pass, then **Submit** passes the hidden tests. Those check agreement with
`scipy.stats.poisson.pmf` on random rates from small to large, including counts above 170. They
also check that the pmf sums to 1 over its support, that the mean and variance are both $\lambda$,
and the exact results at $\lambda = 0$. The families also include probability zero below 0, output
shape that follows `k`, integral floats, invalid `lam` and `k`, and inputs left unchanged.

Some hidden tests take their inputs from the real dataset `online_retail` (see `docs/DATASETS.md`).

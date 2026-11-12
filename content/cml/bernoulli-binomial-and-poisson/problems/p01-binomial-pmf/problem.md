# Binomial probability mass function

Implement `binomial_pmf(k, n, p)`. It returns $P(X=k)$ for each entry of `k`, where
$X \sim \text{Binomial}(n, p)$ counts successes in $n$ independent trials, each with success
probability $p$. It must stay accurate for very large $n$, such as $n = 10^6$. There, the binomial
coefficient $\binom{n}{k}$ is far larger than float64 can hold, and $p^k$ underflows. Work in log space
with the log-gamma function from the standard library.

## Contract

- `k`: integer array-like of any shape. Integral floats such as `2.0` are accepted. Entries outside
  $[0, n]$ have probability `0.0`.
- `n`: integer $n \ge 0$.
- `p`: real number in $[0, 1]$.
- Returns: NumPy `float64` array with the same shape as `k`, holding $P(X=k)$. A scalar `k` gives a
  0-d array.
- Raises `ValueError` when `n` is not a non-negative integer, when `p` is not in $[0, 1]$ (NaN
  included), or when `k` contains a non-integer value.
- Must not mutate its inputs.

## Examples

```python
>>> binomial_pmf([0, 1, 2, 3], 3, 0.5)     # three fair coins: 1/8, 3/8, 3/8, 1/8
array([0.125, 0.375, 0.375, 0.125])        # up to float rounding
>>> binomial_pmf([-1, 11], 10, 0.3)        # outside [0, n] has probability 0
array([0., 0.])
>>> binomial_pmf(0, 10, 0.0)               # p = 0: all mass on k = 0
array(1.)
>>> binomial_pmf(0, 0, 0.5)                # n = 0: the only outcome is 0 successes
array(1.)
>>> binomial_pmf(500000, 10**6, 0.5)       # about 1/sqrt(pi n / 2), about 7.98e-4
array(0.00079788)
>>> binomial_pmf(2.5, 10, 0.3)
ValueError: k must contain integers
```

## Constraints

- Allowed: the standard library `math` (`math.lgamma`, `math.log`, `math.log1p`, …) and NumPy.
  `math.comb` is allowed, but it returns an exact integer that cannot be converted to float for
  large $n$.
- Not allowed: `scipy`. `scipy.stats.binom` would give the answer without the work.

## Done looks like

All public tests pass, then **Submit** passes the hidden tests. Those check agreement with
`scipy.stats.binom.pmf` on random small and medium $n$, and on $n = 10^6$, where the tolerance
is loosened on purpose (see `meta.yaml`). They also check that the pmf sums to 1 over $0..n$, that
the mean is $np$ and the variance $np(1-p)$, and the exact results at $p = 0$, $p = 1$ and $n = 0$.
The families also include probability zero outside $[0, n]$, output shape that follows `k`,
integral floats, the invalid cases for `n`, `p` (NaN included) and `k`, and inputs left unchanged.

Some hidden tests take their inputs from the real dataset `online_retail` (see `docs/DATASETS.md`).

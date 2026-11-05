## Hint 1: concept

The Poisson probability is $\lambda^k e^{-\lambda}/k!$, a power times an exponential over a factorial.
The factorial and the power both overflow for large arguments, but their logarithms do not. Compute
the logarithm of the whole expression, then take `exp` once at the end.

## Hint 2: formula / approach

$\log P(X=k)=k\log\lambda-\lambda-\log\Gamma(k+1)$, where `math.lgamma(k + 1)` gives $\log k!$. Handle
$\lambda = 0$ separately, because $\log 0 = -\infty$. Counts below zero have probability zero, so
the formula is applied only to $k \ge 0$.

## Hint 3: code shape

Validate `lam` with a positive test such as `0.0 <= lam < math.inf`, which rejects NaN too. Convert
`k` to `int64` after the integer check. Use `lgamma = np.vectorize(math.lgamma, otypes=[np.float64])`,
set `out = np.zeros(k_int.shape)`, and write `out[inside] = np.exp(log_pmf)` where
`log_pmf = kk * math.log(lam) - lam - lgamma(kk + 1)` and `kk = k_int[inside]`.

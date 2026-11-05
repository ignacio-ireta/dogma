## Hint 1: concept

The binomial probability is a count of arrangements times the probability of one arrangement:
$\binom nk p^k(1-p)^{n-k}$. For large $n$ neither factor fits in a float. Their logarithms do fit,
and a log of a product is a sum. Convert back with `exp` only at the end.

## Hint 2: formula / approach

$\log P(X=k)=\log\Gamma(n+1)-\log\Gamma(k+1)-\log\Gamma(n-k+1)+k\log p+(n-k)\log(1-p)$, where
`math.lgamma` computes $\log\Gamma$. Handle first the cases where a logarithm is undefined:
$p = 0$ or $p = 1$ (the answer is exact, with no logs), and $k \notin [0, n]$ (the answer is zero).
Validate $n$, $p$ and $k$ before computing anything.

## Hint 3: code shape

`k_arr = np.asarray(k)`, then `k_int = k_arr.astype(np.int64)` after the integer check. Use
`lgamma = np.vectorize(math.lgamma, otypes=[np.float64])` so the formula works elementwise. Build
`out = np.zeros(k_int.shape)`, set `inside = (k_int >= 0) & (k_int <= n)`, compute the log pmf on
`k_int[inside]` only, and write `out[inside] = np.exp(log_pmf)`. Use `math.log1p(-p)` for $\log(1-p)$.

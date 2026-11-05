## Hint 1: concept

Bayes' rule says the posterior is proportional to prior times likelihood, and the evidence only
rescales it to sum to 1. For a sequence of independent observations, the likelihoods multiply. A
long product of probabilities is smaller than the float64 range, so it is safer to add logs.

## Hint 2: formula / approach

For each hypothesis, $\log w_i = \log \pi_i + \sum_t \log L_{i,o_t}$. Then the posterior is
$\exp(\log w_i - \max_k \log w_k)$, normalised to sum to 1. Subtracting the maximum keeps the largest
term at $e^0 = 1$. If the maximum is $-\infty$, every hypothesis has zero evidence and the input is
impossible. Validate first: shapes, non-negative finite entries, sums within $10^{-9}$, integer
observation indices in $[0, o)$.

## Hint 3: code shape

`prior_arr = np.array(prior, dtype=np.float64)`, `lik = np.array(likelihood, dtype=np.float64)`.
Inside `with np.errstate(divide="ignore"):`, compute
`log_post = np.log(prior_arr) + np.log(lik)[:, obs].sum(axis=1)`. Then `peak = log_post.max()`,
raise `ValueError` if `not np.isfinite(peak)`, and return
`w = np.exp(log_post - peak); return w / w.sum()`. Check `obs.min() < 0` as well as
`obs.max() >= o`, since a negative index silently wraps around in NumPy.

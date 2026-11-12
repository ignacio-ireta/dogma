# Beta-Bernoulli maximum likelihood and MAP

A coin has an unknown probability $\theta$ of heads. You observe $k$ heads in $n$ independent flips. Implement
`beta_bernoulli_map(successes, trials, alpha, beta)`, which returns two estimates of $\theta$: the maximum-likelihood
estimate and the maximum-a-posteriori estimate under a $\mathrm{Beta}(\alpha,\beta)$ prior.

## Contract

- `successes` ($k$) and `trials` ($n$): whole numbers with $0 \le k \le n$ and $n \ge 1$. Integers are expected. A
  float with an integral value (such as `3.0`) is accepted. A bool, or a non-integral value such as `2.5`, is not.
- `alpha`, `beta`: finite numbers greater than 0, the shape parameters of the prior.
- Returns: a tuple `(mle, map_estimate)` of two Python floats.
  - `mle` $= k/n$.
  - `map_estimate`: the mode of the posterior $\mathrm{Beta}(a, b)$ with $a = \alpha + k$ and $b = \beta + n - k$, using
    this boundary convention. If $a > 1$ and $b > 1$, the mode is inside $(0,1)$ and equals
    $\dfrac{a-1}{a+b-2}$. If $a \le 1$, the posterior density is largest at $0$, so the mode is $0.0$. Otherwise
    $b \le 1$, the density is largest at $1$, and the mode is $1.0$. These three cases are exhaustive and exclusive
    when $n \ge 1$.
- Raises `ValueError` when `successes` or `trials` is not a whole number, when `trials < 1` (an empty sample has no
  estimate), when `successes` is outside $[0, n]$, or when `alpha` or `beta` is not a finite number greater than 0.

## Examples

```python
>>> beta_bernoulli_map(3, 10, 2, 2)      # posterior Beta(5, 9): mode (5 - 1) / (5 + 9 - 2)
(0.3, 0.3333333333333333)
>>> beta_bernoulli_map(3, 10, 1, 1)      # uniform prior: MAP equals MLE
(0.3, 0.3)
>>> beta_bernoulli_map(0, 5, 0.5, 2)     # no heads, prior below 1 for alpha: mode on the boundary
(0.0, 0.0)
>>> beta_bernoulli_map(5, 5, 1, 0.5)     # all heads, prior below 1 for beta: mode at 1
(1.0, 1.0)
>>> beta_bernoulli_map(0, 0, 1, 1)
ValueError: trials must be at least 1: an empty sample has no maximum likelihood estimate
```

## Constraints

- Allowed: Python arithmetic, `math` and NumPy scalar operations.
- Not allowed: `scipy`. Derive the mode from the posterior density yourself.

## Done looks like

All public tests pass, then **Submit** passes the hidden tests. Those check agreement of the MAP with an independent grid
search over the log-posterior, in the interior case and at both boundaries; the closed form for the interior mode (the
mode, not the posterior mean); the uniform-prior identity that MAP equals MLE for every count; that the MAP always lies
in $[0,1]$; that the MAP approaches the MLE as the data grow; the trial, count and prior validation errors, including fractional
and boolean counts; and that both estimates are Python floats.

The hidden tests draw their bulk inputs from the real dataset `breast_cancer` (see
`docs/DATASETS.md`).

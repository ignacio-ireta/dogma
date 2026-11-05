## Hint 1: concept

The mean is a probability-weighted average of the values. The variance is the probability-weighted
average of the squared distance from that mean, so you need the mean before you can compute it.

## Hint 2: formula / approach

$\mu = \sum_i p_i x_i$ and $\operatorname{Var} = \sum_i p_i (x_i - \mu)^2$. Validate before any
arithmetic: both inputs 1-D with equal length, non-empty, finite, non-negative probabilities, and a sum
within $10^{-9}$ of 1. Test the sum with a tolerance, because float sums are rarely exact.

## Hint 3: code shape

Convert both inputs with `np.asarray(..., dtype=np.float64)`, run the checks in that order, compute
`mean` with a weighted sum, then `deviations = x - mean` and a weighted sum of squared deviations. Take
`math.sqrt` of the variance for the third return value.

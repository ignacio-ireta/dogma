## Hint 1 — concept

Two independent coordinates give a KL that is a sum over coordinates. Each coordinate compares
one pair of one-dimensional Gaussians. Three things matter in each: the log of the scale ratio, the
spread of P under Q's variance, and a constant that makes identical distributions give zero.

## Hint 2 — formula / approach

Use the variances $\sigma^2$ in the spread term. The standard deviations appear only in the log
and in the squared mean difference. For each coordinate compute
$\log(\sigma_q/\sigma_p) + (\sigma_p^2 + (\mu_p-\mu_q)^2)/(2\sigma_q^2) - 1/2$, then sum.

## Hint 3 — code shape

Validate four 1-D arrays of equal length, rejecting non-positive standard deviations. Compute the
three terms as NumPy vectors, combine them, and return `float(np.sum(...))`. Check the result by
hand for two cases: identical arrays should give 0, and a standard-normal P against a N(1, 1) Q
should give 0.5.

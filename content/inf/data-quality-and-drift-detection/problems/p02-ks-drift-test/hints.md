## Hint 1 — concept

The KS statistic is the largest vertical gap between two empirical CDFs. It does not matter which sample is
above the other, so both directions count. A tied block of values must be counted in one step, because a
step taken between two tied values gives the wrong gap.

## Hint 2 — formula / approach

Pool both samples and evaluate both empirical CDFs at every pooled value. The fraction of values at or below x is
the CDF. Take the largest absolute difference as D. For the p-value, the effective size is n * m / (n + m) and
lambda = sqrt(n_eff) * D. The limit is 1 at lambda = 0; otherwise sum the alternating series until its terms are
negligible.

## Hint 3 — code shape

Validate the inputs and alpha first. Sort each sample, then count for every pooled value how many values of each
sample are at or below it, which gives both CDFs in one vectorised step. Compute D, then lambda, then the p-value
with a guard for lambda = 0. Return the statistic, the p-value and the comparison with alpha.

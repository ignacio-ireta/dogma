## Hint 1 — concept

Under the null hypothesis that both arms convert at the same rate, the best estimate of that rate pools
both arms. The standard error of the difference is built from the pooled rate, and the z statistic is the
observed difference divided by that standard error. The p-value is the normal tail probability beyond `|z|`
on both sides.

## Hint 2 — formula / approach

`p_pool = (x_a + x_b) / (n_a + n_b)`, `se = sqrt(p_pool (1 − p_pool) (1/n_a + 1/n_b))`,
`z = (x_b/n_b − x_a/n_a) / se`, and `p = 2 · norm.sf(|z|)`. Check first that the counts are integers in range
and that `p_pool` is strictly between 0 and 1.

## Hint 3 — code shape

Validate the four counts, compute `p_pool`, and raise if it is 0 or 1. Compute `se` and `z` from the arm
rates, then return `(z, 2 * stats.norm.sf(abs(z)))` as floats. Use the survival function, not `1 - cdf`.

## Hint 1: concept

A joint table holds every cell probability. Each marginal is a sum that drops one variable, and
the conditional is the same table restricted to one row, rescaled so that the row sums to 1.
Zero-probability rows cannot be rescaled, so they need their own answer.

## Hint 2: formula / approach

$P(X=i) = \sum_j P_{ij}$ and $P(Y=j) = \sum_i P_{ij}$. Then
$P(Y=j \mid X=i) = P_{ij} / P(X=i)$ for every row where $P(X=i) > 0$. Validate before you
divide: the table is 2-D, finite, non-negative, and sums to 1 within $10^{-9}$.

## Hint 3: code shape

Convert with `np.array(joint, dtype=np.float64)`, check `.ndim`, `np.isfinite(...).all()`,
`(P < 0).any()`, and `abs(P.sum() - 1) > 1e-9`. Then `p_x = P.sum(axis=1)`,
`p_y = P.sum(axis=0)`, and `cond = np.full(P.shape, np.nan)` with
`cond[p_x > 0] = P[p_x > 0] / p_x[p_x > 0, None]`.

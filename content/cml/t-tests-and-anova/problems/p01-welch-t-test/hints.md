## Hint 1 — concept

Welch's test compares two means without assuming the groups share a variance. Each group contributes its
own variance of the mean, `var / n`, and the standard error is the square root of their sum. The degrees of
freedom are not `n_a + n_b − 2`. They come from matching the sum of the two terms to a scaled chi-square
(the Welch–Satterthwaite approximation).

## Hint 2 — formula / approach

Compute `va = var(a, ddof=1) / n_a` and `vb = var(b, ddof=1) / n_b`. Then
`t = (mean(a) − mean(b)) / sqrt(va + vb)` and
`df = (va + vb)**2 / (va**2 / (n_a − 1) + vb**2 / (n_b − 1))`. The p-value is `2 * t.sf(|t|, df)`, with
`t` the Student t distribution from `scipy.stats`. Check the inputs first: 1-D, `n ≥ 2`, finite values, and
a nonzero standard error.

## Hint 3 — code shape

Convert both inputs with `np.asarray(..., dtype=np.float64)` and validate them. Compute the two variance
terms, return `(t, df, p)` as floats, and raise `ValueError` when `va + vb == 0`. If you square tiny
variances directly, `df` can become `nan`: shift and divide by the largest deviation before squaring.

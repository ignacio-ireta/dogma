## Hint 1: concept

Covariance is the average product of two variables' deviations from their means. Each observation
contributes one outer product of its deviation vector, and the average of those outer products is the
matrix.

## Hint 2: formula / approach

$C = (X - \bar{x})^\top (X - \bar{x}) / (n - \text{ddof})$. Subtract the column means before taking any
products. Validate first: 2-D, non-empty, finite, a non-negative integer `ddof`, and $n >
\text{ddof}$, so the divisor is positive.

## Hint 3: code shape

Convert with `np.asarray(X, dtype=np.float64)`, check `.ndim`, `.shape`, `np.isfinite`, `ddof` and
`n`, then compute `centred = x - x.mean(axis=0)` and return `centred.T @ centred / (n - ddof)`. The
subtraction creates a new array, so the input is never changed.

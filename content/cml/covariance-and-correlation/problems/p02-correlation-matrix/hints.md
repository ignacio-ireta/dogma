## Hint 1: concept

Correlation is covariance with each variable's scale divided out. Every entry is the covariance of two
columns divided by the product of their standard deviations, so each variable is measured in units of
its own spread.

## Hint 2: formula / approach

Build the covariance matrix with divisor $n-1$ from the centred columns, then divide entry $[j,k]$ by
$\sqrt{C_{jj} C_{kk}}$. Decide the constant columns first, exactly, by comparing each column's maximum
and minimum. Give those columns NaN in their row and column.

## Hint 3: code shape

Compute `constant = x.max(axis=0) == x.min(axis=0)`. Scale each column by its largest magnitude, centre
the result, form `cov = centred.T @ centred / (n - 1)`, and divide by `np.outer(std, std)` where
`std = np.sqrt(np.diag(cov))`, with `std` replaced by 1 on constant columns to avoid 0/0. Clip to
`[-1, 1]`, then set the constant rows and columns to `np.nan`.

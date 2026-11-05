## Hint 1 — concept

Every entry of the product is a dot product: a row of `A` against a column of `B`. Before any
arithmetic, the shapes must agree. `A` has as many columns as `B` has rows, and the result takes
`A`'s row count and `B`'s column count.

## Hint 2 — formula / approach

`C[i][j] = sum over t of A[i][t] * B[t][j]`. The result has shape (m, p), where `m` comes from `A`
and `p` from `B`. Check `n == k` before computing, and raise `ValueError` with a message that
shows both shapes as tuples. Check that each matrix is a rectangular list of rows, the same way
you did for the transpose problem.

## Hint 3 — code shape

Three nested loops: over the rows of `A` (`i`), over the columns of `B` (`j`), and over the
shared inner index (`t`). Accumulate the sum for each `(i, j)` in a running total, append it to a
row, and collect the rows. Watch the loop bounds: `t` runs over the inner dimension, not over
the rows of `A`.

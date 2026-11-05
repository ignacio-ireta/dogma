## Hint 1: concept

The rank counts the columns that add a new direction. Walk through the columns from left to right.
A column adds a direction if, after removing the contributions of the pivots already chosen, it
still has a non-negligible entry in a row that has not been used as a pivot. Otherwise it is a
combination of earlier columns and adds nothing.

## Hint 2: formula / approach

Keep a counter `rank` and a working copy of `A`. For each column, take the largest-magnitude entry
among rows `rank:`. If that magnitude is at most `tol`, skip the column. Otherwise swap that row into
position `rank`, subtract multiples of it from the rows below, and increment `rank`. The default
`tol` is `1000 * max(m, n) * eps * ||A||_F`, with `||A||_F` computed from the sum of squares, since
`numpy.linalg` is not allowed.

## Hint 3: code shape

`matrix = np.array(A, dtype=np.float64)`, then one loop over columns. Inside the loop:
`pivot_row = rank + int(np.argmax(np.abs(matrix[rank:, col])))`, a `continue` when the pivot is
negligible, a row swap, one outer-product update of the rows below, and `rank += 1`. Stop early when
`rank == rows`.

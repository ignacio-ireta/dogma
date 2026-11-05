## Hint 1: concept

Elimination reduces the matrix to upper triangular form without changing the determinant, except
for one thing: each row swap negates it. Once the matrix is triangular, its determinant is simply
the product of the diagonal. So you only need to track the sign and multiply the pivots.

## Hint 2: formula / approach

`det(A) = sign * product(pivots)`. For each column, choose the largest-magnitude entry on or below
the diagonal, swap it into place (flipping the sign if the row changed), multiply the pivot into the
product, and subtract multiples of the pivot row from the rows below. If the best pivot is exactly
zero, the matrix is singular and the determinant is 0.

## Hint 3: code shape

`matrix = np.array(A, dtype=np.float64)`, then one loop over columns. Inside the loop:
`pivot_row = col + int(np.argmax(np.abs(matrix[col:, col])))`, a `return 0.0` on a zero pivot, a
row swap that flips `sign`, `product *= pivot`, and one outer-product update of the rows below.
Return `float(sign * product)`. The `0 x 0` case falls out of the loop, because the product is
still `1.0`.

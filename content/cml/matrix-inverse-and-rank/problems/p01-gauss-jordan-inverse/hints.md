## Hint 1: concept

Row operations do not change the solution set, so you can apply them to `A` and to `I` at the
same time. When the left side has become `I`, the right side is the inverse. Each column needs
one usable pivot, and a column with no usable pivot means `A` is singular.

## Hint 2: formula / approach

Build `[A | I]`. For each column, choose the row at or below the diagonal with the largest absolute
entry, and swap it into place. If that pivot is at or below `1e-12` times the largest entry of the
original `A`, raise the error. Otherwise scale the pivot row to make the pivot 1, then subtract
multiples of it from every other row so the column becomes a unit vector.

## Hint 3: code shape

`aug = np.hstack([A, np.eye(n)])`. Inside a loop over columns: `pivot_row = col +
int(np.argmax(np.abs(aug[col:, col])))`, a swap with `aug[[col, pivot_row]] = aug[[pivot_row, col]]`,
then `aug[col] /= aug[col, col]`. Clear the column everywhere else with one outer-product update.
Return `aug[:, n:]`. Remember to copy the input first.

## Hint 1 — concept

Transposing swaps the roles of the two indices. The entry in row `i`, column `j` moves to row
`j`, column `i`. So the result has one row per input column, and each of its rows collects one
column of the input.

## Hint 2 — formula / approach

`result[j][i] == matrix[i][j]`, for `j` in the range of the input's column count and `i` in the
range of its row count. The output has `c` rows of length `r`. Before you index anything, check
that every row is a list or tuple and that every row has the same length as row 0. An empty input
returns `[]`, and so does a zero-width input.

## Hint 3 — code shape

After the checks, write a nested comprehension. The outer loop runs over the input's column
indices, and the inner loop runs over the row indices, reading `matrix[row][col]`. Each inner
comprehension builds one new list, so the output never shares rows with the input.

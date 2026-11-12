# Transpose a matrix

Implement `transpose(matrix)`, which returns the transpose of a matrix stored as a nested list.
Row `j` of the result is column `j` of the input, so an input with shape $(r, c)$ gives an output
with shape $(c, r)$. Use plain Python: the point is the index bookkeeping, not NumPy.

## Contract

- `matrix`: a list of rows. Each row is a list or tuple of real numbers (ints allowed). Shape
  $(r, c)$ means $r$ rows, each of length $c$.
  - `[]` is the $(0, 0)$ matrix.
  - Rows of length zero are valid, for example `[[], []]`, shape $(2, 0)$.
- Returns: a new `list` of lists with shape $(c, r)$, where `result[j][i] == matrix[i][j]`.
  Entries are copied unchanged: ints stay ints. The result for any input with $c = 0$ is `[]`,
  because a $(0, r)$ matrix has no rows to record $r$.
- Raises `ValueError` when the rows have different lengths (ragged input), or when `matrix` is not
  a list of rows (for example a flat list such as `[1, 2, 3]`, which is a vector).
- Must not mutate its input. The result must not share row objects with the input.

## Examples

```python
>>> transpose([[1, 2, 3], [4, 5, 6]])
[[1, 4], [2, 5], [3, 6]]
>>> transpose([[1], [2], [3]])          # a column becomes a row
[[1, 2, 3]]
>>> transpose([])
[]
>>> transpose([[1, 2], [3]])
ValueError: ragged rows: row 0 has 2 entries, row 1 has 1
```

## Constraints

- Allowed: plain Python (`len`, `range`, `isinstance`, list comprehensions).
- Not allowed: any `numpy` import. The tests use NumPy as the oracle, not as a tool for you.

## Done looks like

All public tests pass. Then **Submit** runs the hidden tests. They check agreement with NumPy's
transpose on random rectangular matrices, including one-row, one-column and zero-width shapes.
They also check that the shape swaps, that transposing twice returns the input (for matrices with
at least one column), and that float values are copied exactly. Other families: the `ValueError`
cases (ragged rows, flat input), the empty and zero-width results, the return type (lists, not
tuples), and that the input is left unchanged and the result does not share rows with it.

The hidden tests draw their bulk inputs from the real datasets `digits` and `wine` (see
`docs/DATASETS.md`).

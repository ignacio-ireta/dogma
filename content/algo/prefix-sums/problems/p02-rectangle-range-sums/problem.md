# Rectangle range sums

Implement `rectangle_sums(grid, queries)`. Build a padded two-dimensional prefix
table once, then answer each inclusive rectangle with four table lookups.

## Contract

- `grid`: a list of equal-length lists containing Python `int` values. Booleans,
  floats and NumPy integers are excluded. Negative and arbitrarily large integers
  are allowed.
- `queries`: a list of tuples or lists, each holding four Python integers
  `(top, left, bottom, right)`. Booleans are excluded.
- Coordinates are zero-based and **inclusive**:
  `0 <= top <= bottom < rows` and `0 <= left <= right < columns`.
- Returns a list of Python integers in query order. Duplicate queries stay duplicated.
- Empty grids, including zero-width rows, accept only an empty query collection
  and return `[]`.
- Raise `TypeError` for invalid container or element types. Raise `ValueError`
  for ragged rows, a query whose length is not four, or invalid bounds.
- Must not mutate either input.
- Use `O(rows * columns + len(queries))` arithmetic operations and
  `O(rows * columns)` auxiliary storage, excluding the output.

## Examples

```python
rectangle_sums([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [(1, 1, 2, 2), (0, 0, 2, 2), (1, 1, 1, 1)])
# [28, 45, 5]
```

```python
rectangle_sums([[2, -3], [-1, 4]], [(0, 0, 1, 1), (0, 1, 1, 1)])
# [2, 1]
```

## Constraints

Allowed: built-ins and the standard library. Not allowed: `numpy`, `scipy`.
The largest test uses a 300 by 300 grid and 50,000 rectangle queries. Recomputing
each rectangle from its cells will not meet the time budget. Complexity counts
arithmetic operations; arbitrary-precision integers also have a bit-length cost.

## Done looks like

Your results match direct nested sums on randomized small grids, preserve input
and output ordering, handle zero padding and inclusive boundaries, keep large
integer arithmetic exact, reject the stated invalid families, and reuse one
prefix table for a large query batch.

The hidden tests draw their bulk inputs from the real datasets `breast_cancer` and `digits` (see
`docs/DATASETS.md`).

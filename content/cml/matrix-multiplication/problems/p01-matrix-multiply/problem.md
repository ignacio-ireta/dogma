# Matrix multiplication

Implement `matmul(A, B)`, which returns the matrix product $C = AB$ of two matrices stored as
nested lists. Each entry of $C$ is the dot product of a row of `A` with a column of `B`. Before
computing anything, check that the inner dimensions agree, and raise `ValueError` if they do not.
Use plain Python.

## Contract

- `A`: shape $(m, n)$, a list of $m$ rows of length $n$. `B`: shape $(k, p)$, a list of $k$ rows of
  length $p$. Rows are lists or tuples of real numbers (ints allowed). Same representation rules as
  the transpose problem: `[]` is the $(0, 0)$ matrix, and zero-length rows are valid.
- Returns: a list of $m$ rows, each of length $p$, with `C[i][j] = sum over t of A[i][t] * B[t][j]`.
  The shape is $(m, p)$. Entries are numbers: ints when both inputs hold only ints, floats otherwise.
- Raises `ValueError` when:
  - the inner dimensions differ ($n \neq k$). The message must name both shapes as tuples, for
    example `(2, 3) @ (2, 2)`.
  - a matrix is ragged (rows of different lengths), or is not a list of rows.
- Must not mutate its inputs.

## Examples

```python
>>> matmul([[1, 2, 3], [4, 5, 6]], [[7, 8], [9, 10], [11, 12]])
[[58, 64], [139, 154]]
>>> matmul([[1, 0], [0, 1]], [[5, 6], [7, 8]])     # the identity leaves B unchanged
[[5, 6], [7, 8]]
>>> matmul([[1, 2, 3], [4, 5, 6]], [[1, 2], [3, 4]])
ValueError: shape mismatch: (2, 3) @ (2, 2); inner dimensions 3 and 2 differ
```

## Constraints

- Allowed: plain Python (`len`, `range`, `isinstance`, loops or comprehensions).
- Not allowed: any `numpy` import. The tests use NumPy as the oracle, not as a tool for you.

## Done looks like

All public tests pass. Then **Submit** runs the hidden tests. They compare against NumPy's `@` on
random integer and float matrices of several shapes (square, row times column, column times row,
$1 \times 1$). They check algebraic properties: identity on both sides, the transpose of a product,
and associativity on small cases. They also check the entry types: integer inputs give integer
entries, and float inputs give floats. The error families are inner-dimension mismatches (including
pairs whose outer dimensions match, and whose messages must name both shapes), ragged or non-matrix
input, and zero-width input. A few tests target specific slips: an entrywise product instead of the
matrix product, swapped operands, and mutated inputs.

## Hint 1: concept

The projection is the "shadow" of `a` on the line through `b`, so it is always some scalar
multiple `c · b`. Your whole job is finding `c`, plus refusing inputs where no line exists.

## Hint 2: formula / approach

`c = (a·b) / (b·b)`. Choosing `c` this way makes the residual `a - c·b` orthogonal to `b`.
Validate before dividing: both inputs 1-D, equal length, and `b·b != 0`.

## Hint 3: code shape

Convert both inputs with `np.asarray(..., dtype=np.float64)`, check `.ndim` and `.shape`,
compute the two dot products with `@`, and return `coefficient * b_vec`, a new array, so the
inputs stay untouched.

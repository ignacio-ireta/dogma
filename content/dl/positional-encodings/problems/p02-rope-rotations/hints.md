## Hint 1 — concept

Position is the row index along the sequence axis, and each adjacent pair of features is a 2-D
vector that gets turned by an angle. The angle is the position times a frequency, and each pair has
its own frequency. Pair 0 turns fastest, and later pairs turn more slowly.

## Hint 2 — formula / approach

For row `t` and pair `i`, the angle is `t * 10000 ** (-2*i/d)`. A pair `(a, b)` = `(x[2i], x[2i+1])`
turns into `(a*cos - b*sin, a*sin + b*cos)`. Build the angle table with shape `(L, d/2)` by
broadcasting, then apply the 2-D rotation to all pairs at once. Keep the leading batch axes intact.
The width check comes first: `d` must be even and at least 2.

## Hint 3 — code shape

Reshape the input to `(..., L, d//2, 2)` so that `[..., 0]` and `[..., 1]` are the two halves of
each pair. Compute `cos` and `sin` of the `(L, d//2)` angle table, cast them to `x.dtype`, and
combine them with the two halves. Stack the two results along a new last axis and reshape back to
`x.shape`. Never write into `x` itself.

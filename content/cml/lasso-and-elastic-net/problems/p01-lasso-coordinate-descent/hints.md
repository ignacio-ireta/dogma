## Hint 1 — concept

Coordinate descent updates one coefficient at a time. It holds the others fixed and asks for the best
value of this one. With an L1 penalty that best value is a soft-threshold, which is exactly zero when
the coordinate's correlation with the unexplained part of y is small.

## Hint 2 — formula / approach

For coordinate j, let ρⱼ be Xⱼᵀ applied to the residual with wⱼ added back, divided by n, and let zⱼ be
‖Xⱼ‖² divided by n. The new wⱼ is the soft-threshold of ρⱼ at α, divided by zⱼ. Skip any column with
zⱼ = 0. Repeat full sweeps until the largest coefficient change in one sweep is below `tol`.

## Hint 3 — code shape

Keep the residual `y - X @ w` as a running array. Inside the sweep, for each column: add
`column * w[j]` to the residual, compute ρⱼ, compute the new coefficient, subtract `column * new` from the
residual, and track the largest change. Break out of the outer loop after a sweep whose largest change is
below `tol`.

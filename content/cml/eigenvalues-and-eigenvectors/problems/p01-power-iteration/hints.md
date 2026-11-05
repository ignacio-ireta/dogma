## Hint 1: concept

Applying `A` again and again tilts a vector toward the eigenvector whose eigenvalue has the largest
magnitude, because that component grows fastest. Normalise after every step so the numbers stay
bounded. Once the direction has settled, the eigenvalue is the Rayleigh quotient `vᵀAv`, which is
exact for an eigenvector.

## Hint 2: formula / approach

Update `v ← Av / ‖Av‖`. Estimate `λ = vᵀAv`. Stop when `‖Av − λv‖ ≤ tol · max(1, |λ|)`, which is a
residual test, not a test on the change in `λ`. Start from a fixed vector with a nonzero component
along the answer. A seeded random draw is generic enough; a constant vector is not. Check the shape, the finiteness,
the symmetry and the parameters before iterating. At the end, make the first largest-magnitude entry
positive.

## Hint 3: code shape

`v = rng.standard_normal(n)`, then `v = v / sqrt(v @ v)`. Set `eigenvalue = float(v @ (A @ v))`. Loop
over `range(num_iters)`: compute the residual, break if it is small, then `w = A @ v`, break if
`‖w‖ = 0`, and set `v = w / ‖w‖` and update `eigenvalue`. After the loop, find the first index whose
magnitude is within `1e-9` relative of the maximum, and negate `v` if that entry is negative.

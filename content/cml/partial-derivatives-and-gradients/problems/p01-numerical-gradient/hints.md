## Hint 1 — concept

The i-th partial derivative is an ordinary one-variable derivative taken along coordinate `i`, with
every other coordinate frozen. The gradient is the vector of those partials, and it points in the
direction of steepest ascent. A numerical gradient therefore needs one separate one-dimensional
difference per coordinate, and each perturbed point must change exactly one coordinate.

## Hint 2 — formula / approach

For each coordinate `i`, use the central difference
`(f(x + h*e_i) - f(x - h*e_i)) / (2*h)`, where `e_i` is the i-th unit vector and
`h = eps**(1/3) * max(1, abs(x_i))`. Build the two perturbed points as fresh copies of `x`, so the
caller's array is never touched.

## Hint 3 — code shape

Convert `x` with `np.asarray(x, dtype=np.float64)`, then check `ndim == 1`, a non-zero size and
`np.isfinite`. Allocate `grad = np.empty(n)` and loop over the indices. Inside the loop, copy `x`
twice, add and subtract `h` at index `i` in the two copies, and store
`(float(f(plus)) - float(f(minus))) / (2*h)` in `grad[i]`.

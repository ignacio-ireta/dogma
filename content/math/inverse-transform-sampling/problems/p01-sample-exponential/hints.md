## Hint 1 — concept

The inverse CDF turns a uniform draw into an exponential one. If `F(x) = 1 - exp(-rate * x)`, then
`F^-1(u) = -ln(1 - u) / rate`. Each uniform gives one exponential draw, so the whole sample is one
transform of one array of uniforms.

## Hint 2 — formula / approach

Check `rate` and `n` first, and raise `ValueError` before drawing anything. Then call
`u = rng.random(n)` once, and map it with `-np.log1p(-u) / rate`. The `log1p` form computes
`ln(1 - u)` without forming `1 - u` first, so it stays accurate for values of `u` close to 0.

## Hint 3 — code shape

Reject a `rate` that is not finite and positive. Use `math.isfinite(rate) and rate > 0`, which is
also False for NaN. Reject a negative `n`. Draw the array once, transform it as a whole with NumPy,
and return the result. The output of `rng.random(0)` is already a valid empty float64 array.

## Hint 1 — concept

Linear position interpolation does not change the rotation itself. It changes the position fed into
the rotation: every position is compressed by the same factor, so a longer window reuses the angles the
model has already seen. The base stays at 10000.

## Hint 2 — formula / approach

Use the scaled position $s_t = (\mathrm{offset}+t)/\mathrm{scale}$, and the usual frequencies
$\theta_i = 10000^{-2i/d}$. The angle is $\phi_{t,i} = s_t\,\theta_i$. The rotation of a half-split pair
is the same as in the core problem. Divide the offset together with $t$, not after the rotation.

## Hint 3 — code shape

Build the scaled positions as one float64 vector, `(offset + arange(T)) / scale`, and reshape it to
`(T, 1)`. Multiply by the `(d/2,)` frequency vector to get a `(T, d/2)` angle table. Take `cos` and
`sin`, then reuse the core problem's split, rotate and join steps for q and k. Validate `scale` as a
finite number greater than zero before any arithmetic.

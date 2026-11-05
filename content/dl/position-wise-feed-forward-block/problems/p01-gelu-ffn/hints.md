## Hint 1 — concept

The block is two linear maps with a nonlinearity between them, and the same map is applied to
every position. Positions are the leading axes; the features are the last one. Nothing crosses
positions, so the batch and sequence axes just ride along.

## Hint 2 — formula / approach

Expansion: `z = x @ W1.T + b1`, which gives `d_ff` features. Then apply the exact GELU, which is
`z * Phi(z)` with `Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))`. Then project back with
`h @ W2.T + b2`. Check the shapes first. `W1` is `(d_ff, d_model)`, and `W2` must be
`(d_model, d_ff)`.

## Hint 3 — code shape

Unpack `d_ff, d_model = W1.shape` after checking that `W1` is 2-D. Compare `X.shape[-1]` with
`d_model`. Compare `W2.shape`, `b1.shape` and `b2.shape` with their tuples. Compute `z`, then the
GELU using `torch.erf` and `math.sqrt(2.0)`, then the output projection. Do not use `torch.nn`.

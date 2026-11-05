## Hint 1 — concept

A pre-norm block normalises the input of each sub-layer and adds the sub-layer output to the
un-normalised stream. The attention sub-layer must not let a token see later tokens.

## Hint 2 — formula / approach

Write four helpers: LayerNorm over the last axis with biased variance, the exact GELU through
`erf`, a head split and merge, and a causal masked softmax. Then compose the block as two
residual steps. Use `x' = x + Attn(LN1(x))` before the second sub-layer, and
`output = x' + FFN(LN2(x'))`.

## Hint 3 — code shape

Project each of q, k, v with `h @ W + b`. Reshape `(B, T, D)` to `(B, T, H, d_head)`, then
transpose to `(B, H, T, d_head)`. Compute scores with `q @ k.transpose(-2, -1)` and divide by
`sqrt(d_head)`. Mask the strict upper triangle with `masked_fill(..., float("-inf"))`, apply
`softmax(dim=-1)`, multiply by v, and undo the transpose and reshape.

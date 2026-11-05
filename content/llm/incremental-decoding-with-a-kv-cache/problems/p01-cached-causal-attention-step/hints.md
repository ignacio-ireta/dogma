## Hint 1 — concept

Only the new tokens need projecting, because the cache already holds the keys and values of earlier
tokens. The new queries must still see all of them. A query at position `S + i` has seen `S + i + 1`
tokens, so the mask depends on where the chunk starts, not only on its length.

## Hint 2 — formula / approach

Project `x_new` to `Q`, `K_new` and `V_new`, then split into heads with a reshape and a transpose.
Concatenate `K_new` after `K_cache` along the key axis, and the same for `V`. Scores are
`Q K^T / sqrt(d_k)`, of shape `(B, h, T, S + T)`. Mask entry `(i, j)` is allowed when `j <= S + i`.
Softmax, weight the values, merge heads, and apply `W_o`. Validate the arguments before any
arithmetic.

## Hint 3 — code shape

Build a boolean mask of shape `(T, S + T)` in which row `i` is allowed up to and including column
`S + i`, and compare column indices with row indices plus `S`. Use
`scores.masked_fill(~allowed, float("-inf"))` before the softmax. For the cache, use
`torch.cat([K_cache, K_new_heads], dim=2)` when the cache is not `None`, and return
`(output, (K_all, V_all))`.

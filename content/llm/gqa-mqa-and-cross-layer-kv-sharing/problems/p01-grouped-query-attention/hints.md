## Hint 1 — concept
Attention has one query per head, but grouped-query attention stores fewer key and value heads. Several query
heads then share one key head and one value head. Only the number of K and V heads changes. Each query head
still computes its own softmax over the keys.

## Hint 2 — formula / approach
With `group_size = n_heads / n_kv_heads`, query head `h` reads key and value head `h // group_size`. Project
Q to `d_model` and K, V to `n_kv_heads * d_head`. Expand K and V so that each query head has its own copy of its
group's head, then run ordinary per-head attention with scale `1/sqrt(d_head)` and merge the heads.

## Hint 3 — code shape
Reshape Q to `(B, L, n_heads, d_head)` and K, V to `(B, L, n_kv_heads, d_head)`, then move the head axis in front
of the token axis. Expand K and V along the head axis with a repeat that places each key/value head in a
contiguous block of `group_size` copies. Check the copy order against the one-token example in problem.md. The
rest is the same as multi-head attention: scores, mask, softmax, weighted sum, merge, output projection.

# Grouped-query attention forward pass

Implement `grouped_query_attention(X, W_q, W_k, W_v, W_o, n_heads, n_kv_heads, mask=None)`. It runs
one self-attention layer in which the query heads outnumber the key and value heads. Every query head
still gets its own attention pattern, but the query heads are split into `n_kv_heads` groups, and each
group shares one key head and one value head.

The layer has no biases. Its only parameters are the four matrices you are given.

Let `d_model` be the width of `X`, `d_head = d_model // n_heads` the width of one head, and
`group_size = n_heads // n_kv_heads` the number of query heads per group. **Query head `h` (from 0 to
`n_heads - 1`) uses key and value head `h // group_size`.** The groups are contiguous: query heads
`0 .. group_size - 1` share key/value head 0, the next `group_size` heads share key/value head 1, and so on.

The attention in query head `h` is `softmax(Q_h K_g^T / sqrt(d_head) + mask) V_g`, where `g = h // group_size`.
Then the heads are concatenated in order and multiplied by `W_o`.

## Contract

- `X`: `torch.float64`, shape `(B, L, d_model)`. `B >= 1`, `L >= 1`.
- `W_q`: `torch.float64`, shape `(d_model, d_model)`, in PyTorch `nn.Linear` layout: a projection is
  `y = x @ W.T`. Its output features are the `n_heads` query heads in order.
- `W_k`, `W_v`: `torch.float64`, shape `(n_kv_heads * d_head, d_model)`, also `nn.Linear` layout. Output
  feature block `g` (rows `g*d_head .. (g+1)*d_head - 1`) is key or value head `g`.
- `W_o`: `torch.float64`, shape `(d_model, d_model)`, `nn.Linear` layout. It maps the concatenated heads
  to `d_model`.
- `n_heads`, `n_kv_heads`: Python `int`s. `n_kv_heads >= 1`, `n_heads >= n_kv_heads`, and `n_kv_heads`
  divides `n_heads`. `n_heads` divides `d_model`.
- `mask`: `None`, or a `torch.bool` tensor of shape `(L, L)`. `mask[i, j]` is `True` when query token `i`
  may attend to key token `j`. The mask is shared by all sequences and all heads.
- Returns: `torch.float64`, shape `(B, L, d_model)`.
- Raises `ValueError` when: `X` is not 3-D; any weight has the wrong shape (for example, a full-width key
  projection when `n_kv_heads < n_heads`); `n_heads` or `n_kv_heads` is not a positive integer or is a
  `bool`; `n_kv_heads` does not divide `n_heads`; `n_heads` does not divide `d_model`; `mask` is not a
  `torch.bool` tensor of shape `(L, L)`; or some query row of `mask` has no `True` entry.
- Must not mutate inputs.

## Examples

```python
>>> # One token, d_model = 4, 4 query heads, 2 key/value heads, so group_size = 2 and d_head = 1.
>>> X = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]], dtype=torch.float64)          # (1, 1, 4)
>>> W_q = torch.eye(4, dtype=torch.float64)
>>> W_k = torch.tensor([[1.0, 0, 0, 0], [0, 1.0, 0, 0]], dtype=torch.float64)  # keys:   [1, 2]
>>> W_v = torch.tensor([[0, 0, 1.0, 0], [0, 0, 0, 1.0]], dtype=torch.float64)  # values: [3, 4]
>>> W_o = torch.eye(4, dtype=torch.float64)
>>> grouped_query_attention(X, W_q, W_k, W_v, W_o, n_heads=4, n_kv_heads=2)[0, 0]
tensor([3., 3., 4., 4.], dtype=torch.float64)
```

With one token the softmax has one key, so each head outputs its key/value head's value. Query heads 0
and 1 read value head 0 (value 3), and query heads 2 and 3 read value head 1 (value 4). The modulo mapping
`h % n_kv_heads` would give `[3, 4, 3, 4]`.

```python
>>> # n_kv_heads == n_heads is ordinary multi-head attention; a 3-token mask example is in the hidden tests.
>>> grouped_query_attention(X, W_q, W_k, W_v, W_o, n_heads=4, n_kv_heads=3)
ValueError: n_kv_heads must divide n_heads
```

## Constraints

- Allowed: the `torch` package for tensor operations, for example `torch.matmul`, `torch.softmax`,
  `reshape`, `transpose`, `repeat_interleave`, `index_select`, `masked_fill`, and the `math` module
  for `sqrt`.
- Not allowed: `torch.nn` in any form (for example `torch.nn.functional.scaled_dot_product_attention`,
  `torch.nn.MultiheadAttention` and `torch.nn.Linear`), and the top-level `torch.scaled_dot_product_attention`.
  They are the test oracles, not the tool. Write the projections, the head grouping, the softmax and the
  output projection yourself.

## Done looks like

All public tests pass. Then Submit runs the hidden tests. They cover: agreement with an oracle that uses
`torch.nn.functional.scaled_dot_product_attention` on seeded random inputs, with an explicit
key/value index per query head (several head counts, including multi-query with one key/value head and
the case with `n_kv_heads == n_heads`); the head mapping itself, checked on a one-token example where the
modulo mapping gives a different answer; the mask meaning (`True` is allowed) and the fully masked-row
error; the `1/sqrt(d_head)` scale, checked where it differs from `1/sqrt(d_model)`; the weight layout on
non-symmetric weights; the shape and `ValueError` rules, including a full-width key projection and a head
count that does not divide; batch independence; large and small input magnitudes; dtype and shape; and
inputs left unchanged.

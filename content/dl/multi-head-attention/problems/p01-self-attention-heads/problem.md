# Multi-head self-attention forward pass

Implement `multi_head_self_attention(X, W_q, W_k, W_v, W_o, b_q, b_k, b_v, b_o, num_heads, mask=None)`.
It runs one multi-head self-attention layer on a batch of sequences. The steps are: project the
input to queries, keys and values; split the features into `num_heads` heads; run scaled
dot-product attention inside each head; concatenate the heads; apply the output projection.
Every weight is given to you explicitly. Do not create parameters.

Per head `j`, with `d_k = d_model / num_heads`, the attention is
`softmax(Q_j K_j^T / sqrt(d_k) + mask_penalty) V_j`, where `Q_j`, `K_j`, `V_j` are the columns of
`Q`, `K`, `V` that belong to head `j`.

## Contract

- `X`: `torch.float64` tensor, shape `(B, L, d_model)`. `B >= 1` sequences, `L >= 1` tokens each.
- `W_q`, `W_k`, `W_v`, `W_o`: `torch.float64`, shape `(d_model, d_model)`, in **PyTorch
  `nn.Linear` layout**: a projection is `y = x @ W.T + b`. Row `o` of `W` holds the weights of output
  feature `o`. `W_o` projects the concatenated heads back to `d_model`.
- `b_q`, `b_k`, `b_v`, `b_o`: `torch.float64`, shape `(d_model,)`.
- `num_heads`: Python `int`, `>= 1`. It must divide `d_model` exactly.
- `mask`: `None`, or a `torch.bool` tensor of shape `(L, L)`. Entry `[i, j]` is `True` when query
  token `i` may attend to key token `j`, and `False` when it may not. The same mask applies to every
  sequence and every head. A `False` entry gets zero attention weight. `True` means "allowed", the
  opposite of `torch.nn.MultiheadAttention`.
- Returns: `torch.float64` tensor of shape `(B, L, d_model)`.
- Raises `ValueError` when: `X` is not 3-D; any weight is not `(d_model, d_model)`; any bias is not
  `(d_model,)`; `num_heads` is not a positive integer that divides `d_model`; `mask` is given but is
  not a `torch.bool` tensor of shape `(L, L)`; or some query row of `mask` has no `True` entry (its
  softmax would be undefined).
- Must not mutate inputs.

## Examples

```python
>>> I = torch.eye(2, dtype=torch.float64); z = torch.zeros(2, dtype=torch.float64)
>>> X = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float64)   # B=1, L=2, d_model=2
>>> out = multi_head_self_attention(X, I, I, I, I, z, z, z, z, num_heads=1)
>>> out[0, 0]  # row 0 attends to rows 0 and 1 with weights softmax([1/sqrt2, 0])
tensor([0.6698, 0.3302], dtype=torch.float64)
```

```python
>>> mask = torch.tensor([[True, False], [True, True]])   # token 0 sees only itself
>>> out = multi_head_self_attention(X, I, I, I, I, z, z, z, z, num_heads=1, mask=mask)
>>> out[0, 0]  # now row 0 is exactly its own value vector
tensor([1., 0.], dtype=torch.float64)
```

```python
>>> multi_head_self_attention(X, I, I, I, I, z, z, z, z, num_heads=3)
ValueError: num_heads must divide d_model
```

## Constraints

- Allowed: the `torch` package, for example `torch.matmul`, `torch.softmax` (allowed here; the scaled dot-product problem bans it), `torch.reshape`,
  `transpose`, `masked_fill` and the `math` module for `sqrt`.
- Not allowed: `torch.nn` in any form, including `torch.nn.MultiheadAttention`,
  `torch.nn.Linear` and `torch.nn.functional.scaled_dot_product_attention`. They are the test
  oracles, not the tool. Write the projections, head split, softmax and output projection yourself.

## Done looks like

All public tests pass. Then **Submit** runs the hidden tests. They cover: agreement with a
PyTorch `MultiheadAttention` loaded with the same weights, on seeded random batches with one, two
and four heads, with and without masks; the per-head feature blocks (heads must not mix); the
`1/sqrt(d_k)` scale, checked at a head width where it differs from `1/sqrt(d_model)`; the weight
layout (the `x @ W.T` convention, on non-symmetric weights); the mask meaning (`True` is allowed)
and the fully-masked-row error; the output bias added once; the `ValueError` rules (rank, weight
and bias shape, head count that does not divide, mask dtype and shape); batch independence; large
and small input magnitudes; the dtype and shape; and inputs left unchanged.

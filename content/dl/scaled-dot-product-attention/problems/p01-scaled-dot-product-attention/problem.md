# Masked scaled dot-product attention

Implement `scaled_dot_product_attention(q, k, v, mask=None)`. It computes
`softmax(q @ k^T / sqrt(d_k)) @ v` over the last two axes, applies an optional boolean mask
that says which query-key pairs may attend, and returns both the output and the attention
weights. Leading batch axes are shared by `q`, `k` and `v`.

## Contract

- `q`: floating-point tensor of shape `(*batch, L, d_k)`.
- `k`: floating-point tensor of shape `(*batch, S, d_k)`.
- `v`: floating-point tensor of shape `(*batch, S, d_v)`.
- `q`, `k` and `v` have the same dtype and identical `batch` shapes. All of `L`, `S`, `d_k`, `d_v`
  and every batch axis are at least 1. Each tensor has rank at least 2.
- `mask`: `None`, or a boolean tensor that `torch.broadcast_to` can expand to `(*batch, L, S)`.
  `True` means the query may attend the key; `False` means it may not.
- Scores: `z = q @ k^T / sqrt(d_k)`, shape `(*batch, L, S)`. Disallowed pairs are excluded
  from the softmax, so their weight is exactly `0`.
- Weights: softmax over the key axis of `z`, shape `(*batch, L, S)`, same dtype as `q`. For every
  query row with at least one allowed key, the allowed weights sum to 1.
- Fully masked rows: a query row with no allowed key has weight `0` on every key and output `0`.
  No `NaN` and no exception for such a row. Other rows are unaffected.
- Output: `weights @ v`, shape `(*batch, L, d_v)`, same dtype as `q`.
- Returns: a tuple `(output, weights)`.
- Raises: `ValueError` when a tensor is not floating point, dtypes differ, ranks are below 2,
  batch shapes differ, any size is zero, the mask is not boolean or cannot be broadcast to
  `(*batch, L, S)`.
- Must be finite for finite inputs whose scores `q @ k^T / sqrt(d_k)` are finite, including scores
  far outside the range of `exp`. Must not mutate inputs.

## Examples

```python
>>> q = torch.tensor([[1.0, 0.0]])            # L=1, d_k=2
>>> k = torch.eye(2)                          # S=2
>>> v = torch.tensor([[10.0, 0.0], [0.0, 10.0]])
>>> out, w = scaled_dot_product_attention(q, k, v)
>>> w                                         # scores [1, 0] / sqrt(2)
tensor([[0.6698, 0.3302]])
>>> out
tensor([[6.6976, 3.3024]])
```

```python
>>> mask = torch.tensor([[True, False]])      # query 0 may attend key 0 only
>>> out, w = scaled_dot_product_attention(q, k, v, mask)
>>> w
tensor([[1., 0.]])
>>> out
tensor([[10., 0.]])
```

```python
>>> mask = torch.tensor([[False, False]])     # no allowed key: fully masked row
>>> out, w = scaled_dot_product_attention(q, k, v, mask)
>>> w, out
(tensor([[0., 0.]]), tensor([[0., 0.]]))
```

## Constraints

- Allowed: torch tensor operations, including `torch.exp`, `torch.amax`, `sum`, `masked_fill`,
  `torch.where`, matrix multiplication and `transpose`.
- Not allowed: `torch.nn` (including `torch.nn.functional`, which provides attention and softmax)
  and `torch.softmax` (also reached as `x.softmax(...)`). Write the max-subtracted softmax yourself.

## Done looks like

The three examples pass, including the fully masked row. Hidden tests compare seeded random
batched inputs with a `torch.nn.functional` oracle, check that every weight row sums to 1 and
that masked pairs have zero weight, check the `sqrt(d_k)` scale at `d_k = 64`, and stress large
logits (queries scaled by `1e3`) with and without masks. They also cover fully masked rows,
broadcastable masks, boolean-only masks, dtype, zero-size and batch-shape errors, and input
non-mutation.

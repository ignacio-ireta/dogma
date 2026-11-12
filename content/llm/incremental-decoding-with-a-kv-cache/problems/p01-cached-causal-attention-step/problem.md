# Cached causal attention step

Implement `cached_attention_step(x_new, cache, W_q, W_k, W_v, W_o, num_heads)`. It runs one
multi-head self-attention layer for a chunk of `T` new tokens, given the keys and values of all
earlier tokens in `cache`. It returns the output for the new tokens and the extended cache. A
prefill call is this step with no cache and `T` equal to the prompt length. A decode call uses
`T = 1`. The result must equal full causal attention over the whole prefix, restricted to the new
rows.

## Contract

- `x_new`: `torch.float64` tensor of shape `(B, T, d)`, with `B >= 1` and `T >= 1`.
- `cache`: either `None` (an empty cache), or a tuple `(K_c, V_c)` of two `torch.float64` tensors,
  each of shape `(B, h, S, d_k)`, with `S >= 0`. `K_c` and `V_c` hold the keys and values of the
  `S` earlier tokens, already split into heads, with `d_k = d // h`.
- `W_q`, `W_k`, `W_v`, `W_o`: `torch.float64` tensors of shape `(d, d)` in `nn.Linear` layout, so a
  projection is `y = x @ W.T`. There are no biases.
- `num_heads`: `int`, `>= 1`. It must divide `d`. A `bool` is rejected.
- Steps:
  1. Project only the new tokens: `Q`, `K_new` and `V_new`, each `(B, T, d)`, then split them into
     heads of shape `(B, h, T, d_k)`.
  2. Append `K_new` and `V_new` after the cache along the key axis, so `K` and `V` have shape
     `(B, h, S + T, d_k)`.
  3. Score every new query against every key with `Q K^T / sqrt(d_k)`. The new query at row `i`
     (`0 <= i < T`) sits at absolute position `S + i`. It may attend key `j` exactly when
     `j <= S + i`. Masked scores get zero weight.
  4. Softmax over the key axis, take the weighted sum of `V`, merge the heads back to `(B, T, d)`,
     and apply `W_o`.
- Returns: a tuple `(output, new_cache)`. `output` is `torch.float64` of shape `(B, T, d)`.
  `new_cache` is a tuple `(K, V)` of `torch.float64` tensors, each `(B, h, S + T, d_k)`, whose first
  `S` positions equal the input cache exactly.
- Raises `ValueError` when: `x_new` is not a 3-D `float64` tensor; `B = 0` or `T = 0`; `d` is not
  divisible by `num_heads`; `num_heads` is not a positive `int`; a weight is not `(d, d)` float64;
  `cache` is neither `None` nor a 2-element tuple or list; a cache tensor is not a 4-D `float64`
  tensor of shape `(B, h, S, d_k)` with `K` and `V` the same shape.
- Must not mutate `x_new`, the weights, or the input cache tensors.

## Examples

Identity weights, `d = 2`, `num_heads = 1`. `I` is the `2 x 2` identity.

```python
>>> I = torch.eye(2, dtype=torch.float64)
>>> out, cache = cached_attention_step(torch.tensor([[[1.0, 0.0]]], dtype=torch.float64), None, I, I, I, I, 1)
>>> out
tensor([[[1., 0.]]], dtype=torch.float64)
>>> cache[0].shape            # K is (B, h, S + T, d_k) = (1, 1, 1, 2)
torch.Size([1, 1, 1, 2])
```

```python
>>> out, cache = cached_attention_step(torch.tensor([[[0.0, 1.0]]], dtype=torch.float64), cache, I, I, I, I, 1)
>>> out                       # query [0, 1] scores the two keys [0, 0.7071] (scaled by 1/sqrt(2))
tensor([[[0.3302, 0.6698]]], dtype=torch.float64)
```

```python
>>> x = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float64)  # a two-token chunk
>>> out, _ = cached_attention_step(x, None, I, I, I, I, 1)
>>> out                       # row 0 sees only itself; row 1 sees both tokens
tensor([[[1.0000, 0.0000],
         [0.3302, 0.6698]]], dtype=torch.float64)
```

## Constraints

- Allowed: `torch` tensor operations such as `matmul`, `reshape`, `transpose`, `cat`, `ones`,
  `tril`, `masked_fill`, `softmax` and `exp`; the `math` module for `sqrt`; Python `int`, `bool`
  and `tuple` checks.
- Not allowed: `torch.nn` in any form. That includes `torch.nn.MultiheadAttention`,
  `torch.nn.functional.scaled_dot_product_attention` and any `is_causal` flag, because those are the
  test oracles, not the tool. Build the bottom-right mask yourself.

## Done looks like

All public tests pass. Then **Submit** runs the hidden tests. They cover: agreement with full causal
recomputation over the whole prefix, at each of several decode steps and a prefill chunk, for one and
several heads; the bottom-right offset once the cache is non-empty; a multi-token chunk that is
causal within itself; the cache growing by exactly `T` positions and keeping earlier positions
unchanged; the input cache left unchanged; batch independence; large and small magnitudes; the
empty-cache path matching the `None` path; the output dtype and shape; and the `ValueError` rules for
the input, weight, head count and cache shapes.

The hidden tests draw their bulk inputs from the real dataset `breast_cancer` (see
`docs/DATASETS.md`).

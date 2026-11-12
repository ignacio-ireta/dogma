# Pre-norm causal decoder block

Implement `pre_norm_block(x, params, num_heads)`, the forward pass of one pre-norm decoder block.
The block has two residual sub-layers, each applied after a LayerNorm of its input:

1. causal multi-head self-attention, then add the result to the stream;
2. a position-wise feed-forward network with exact GELU, then add the result to the stream.

Normalise the **input** of each sub-layer, never the sum that the residual produces. Use
row-vector projections, `y = x @ W + b`, with `W` stored as `(in, out)`, the layout of the
dense-layer exercise. There is no dropout: the block is evaluated deterministically.

## Contract

- `x`: floating-point tensor of shape `(B, T, D)`, with `B, T, D >= 1`. `B` is the batch, `T` the
  number of tokens, `D` the model width.
- `num_heads`: positive `int` (not `bool`) that divides `D`. Each head has width `d_head = D / num_heads`.
- `params`: a mapping with **exactly** these 16 keys, no more and no fewer. `F >= 1` is the
  feed-forward width, read from `W_1`.

  | key | shape | role |
  |---|---|---|
  | `ln1_gamma`, `ln1_beta` | `(D,)` | scale and shift of the attention-block LayerNorm |
  | `W_q`, `W_k`, `W_v`, `W_o` | `(D, D)` | query, key, value and output projections |
  | `b_q`, `b_k`, `b_v`, `b_o` | `(D,)` | their biases |
  | `ln2_gamma`, `ln2_beta` | `(D,)` | scale and shift of the feed-forward-block LayerNorm |
  | `W_1` | `(D, F)` | feed-forward expansion |
  | `b_1` | `(F,)` | its bias |
  | `W_2` | `(F, D)` | feed-forward projection back to the model width |
  | `b_2` | `(D,)` | its bias |

  Every parameter has the same dtype as `x`.
- Definitions, with $\varepsilon = 10^{-5}$:
  - LayerNorm over the last axis, with **biased** variance: $\mathrm{LN}(u) = \gamma \odot (u-\mu)/\sqrt{\sigma^2+\varepsilon}+\beta$,
    where $\mu$ and $\sigma^2$ are the mean and the variance over the $D$ features, dividing by $D$.
  - Exact GELU: $\mathrm{gelu}(z) = \tfrac12 z\,(1+\operatorname{erf}(z/\sqrt2))$. The tanh approximation is not allowed.
  - Attention: $q,k,v$ are the projections of $\mathrm{LN}_1(x)$. Split features into heads by
    contiguous slices. Scores are $q\cdot k/\sqrt{d_{\text{head}}}$. A **causal** mask forbids
    query position $i$ from attending to any key $j>i$. Softmax runs over keys. Concatenate
    the heads and apply $W_o$ and $b_o$.
- Computation, in this order:
  1. $h = \mathrm{LN}_1(x)$; attention output $a = \mathrm{Attn}(h)$; $x' = x + a$.
  2. $h' = \mathrm{LN}_2(x')$; $f = \mathrm{gelu}(h' W_1 + b_1)\,W_2 + b_2$; output $= x' + f$.
- Returns: `(B, T, D)` tensor, same dtype as `x`.
- Raises `ValueError` when: `x` is not a 3-D floating tensor, or has a zero-size axis;
  `num_heads` is not a positive `int` or does not divide `D`; `params` is missing a key, has an
  extra key, or has a value that is not a tensor; any parameter has the wrong shape; or any
  parameter's dtype differs from `x`.
- Must not mutate `x` or any parameter.

## Examples

```python
>>> x = torch.tensor([[[1.0, 2.0]]], dtype=torch.float64)           # B=1, T=1, D=2
>>> # every projection is zero, except gamma = 1 in both LayerNorms
>>> pre_norm_block(x, params, num_heads=1)
tensor([[[1., 2.]]], dtype=torch.float64)    # both sub-layers add zero: the block is the identity
```

```python
>>> x = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float64)  # B=1, T=2, D=2
>>> # W_q = W_k = W_v = W_o = W_1 = I, W_2 = 0, b = 0, gamma = 1, beta = 0, num_heads = 1
>>> pre_norm_block(x, params, num_heads=1)
tensor([[[ 2.0000, -1.0000],
         [-0.8884,  1.8884]]], dtype=torch.float64)
# Token 0 sees only itself. Token 1 weights token 0 by about 0.056 and itself by about 0.944.
```

```python
>>> pre_norm_block(x, params, num_heads=3)   # D = 2 is not divisible by 3
ValueError: num_heads must divide D so each head gets D / num_heads features
```

## Constraints

- Allowed: PyTorch tensor operations such as `matmul`, `reshape`, `transpose`, `softmax`,
  `masked_fill`, `triu`, `erf`, `sqrt` and `torch.is_tensor`, plus the standard library (`math`).
- Not allowed: `torch.nn`, including `torch.nn.functional`, `torch.nn.LayerNorm`,
  `torch.nn.MultiheadAttention` and `torch.nn.TransformerEncoderLayer`. Those are the test
  oracles, not the tool. Write LayerNorm, the head split, the mask and GELU yourself.

## Done looks like

All public tests pass, then **Submit** runs the hidden tests. They check: agreement with an
independent reference built from `torch.nn.functional` pieces on seeded random batches of several
shapes; agreement with `torch.nn.TransformerEncoderLayer` (pre-norm, GELU, no dropout) loaded
with the same weights; no information from future tokens reaching earlier positions; the
identity behaviour when the sub-layers are zero; large input magnitudes; the exact GELU and the
biased variance; the input and parameters left unchanged; the output shape and dtype; and every
`ValueError` rule above. When an output matches a known wrong variant, the failure names that
mistake, for example post-norm placement, a missing mask, or a missing residual.

The hidden tests draw their bulk inputs from the real dataset `breast_cancer` (see
`docs/DATASETS.md`).

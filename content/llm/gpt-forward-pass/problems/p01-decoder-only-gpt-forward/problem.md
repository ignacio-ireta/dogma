# Decoder-only GPT forward pass

Implement `gpt_forward(idx, params, num_heads)`, the forward pass of a decoder-only language
model. It maps integer token ids to next-token logits in five steps:

1. **Embed.** Look up each token's embedding and add the embedding of its position.
2. **Blocks.** Run the residual stream through every pre-norm causal block, in order.
3. **Final LayerNorm.** Normalise the stream once more before the output head.
4. **LM head.** Project each position onto the vocabulary. The head is **tied** to the token
   embedding when `params` has no `lm_head` key. Otherwise it uses `lm_head`.
5. **Logits.** Return one score per vocabulary entry for every position.

Each block is the same pre-norm block as in `dl/transformer-block`, with the same 16 keys and the
same shapes. Write that block again in your file. You may not import it from anywhere. Use
row-vector projections `y = x @ W + b`, with every weight stored as `(in, out)`.

## Contract

- `idx`: `torch.int64` tensor of shape `(B, T)`, with `B >= 1` and `T >= 1`. Entry `(b, t)` is the
  token id at position `t` of row `b`. Ids lie in `[0, V)`.
- `num_heads`: positive `int` (not `bool`) that divides `D`. Each head has width `D / num_heads`.
- `params`: a mapping with these keys. `V`, `D`, `C` and the block feed-forward width `F` are read
  from the shapes given here.

  | key | shape | role |
  |---|---|---|
  | `tok_emb` | `(V, D)` | token embedding table |
  | `pos_emb` | `(C, D)` | position embedding table; row `t` is position `t` |
  | `blocks` | list of `N` mappings, `N >= 0` | one block per entry, each with the 16 keys below |
  | `ln_f_gamma`, `ln_f_beta` | `(D,)` | final LayerNorm scale and shift |
  | `lm_head` | `(D, V)` | **optional.** Present means untied. Absent means tied |

  Each block mapping has exactly these 16 keys, with the shapes of `dl/transformer-block`:
  `ln1_gamma`, `ln1_beta` `(D,)`; `W_q`, `W_k`, `W_v`, `W_o` `(D, D)`; `b_q`, `b_k`, `b_v`, `b_o`
  `(D,)`; `ln2_gamma`, `ln2_beta` `(D,)`; `W_1` `(D, F)`; `b_1` `(F,)`; `W_2` `(F, D)`; `b_2` `(D,)`.
  Each block's feed-forward width `F >= 1` is read from its own `W_1`.

  Every tensor in `params` has the dtype of `tok_emb`, which is a floating-point dtype.
- Definitions, with $\varepsilon = 10^{-5}$:
  - Embedding: $x_{b,t} = \mathrm{tok\_emb}[idx_{b,t}] + \mathrm{pos\_emb}[t]$. Shape $(B, T, D)$.
  - LayerNorm over the last axis with **biased** variance, $\mathrm{LN}(u)=\gamma\odot(u-\mu)/\sqrt{\sigma^2+\varepsilon}+\beta$.
  - Exact GELU: $\mathrm{gelu}(z)=\tfrac12 z\,(1+\operatorname{erf}(z/\sqrt2))$.
  - Causal attention: query $i$ may not attend to any key $j>i$. Scores are scaled by $1/\sqrt{d_\text{head}}$.
  - Tied head: logits $=h\,\mathrm{tok\_emb}^\top$, with $h$ the final LayerNorm output.
  - Untied head: logits $=h\,\mathrm{lm\_head}$.
- Returns: `(B, T, V)` tensor, the logits. Same dtype as `tok_emb`.
- Raises `ValueError` when: `idx` is not a 2-D `int64` tensor, or has a zero-size axis; any id is
  negative or at least `V`; `T > C`; `num_heads` is not a positive `int` or does not divide `D`;
  `params` is not a mapping, is missing a required key, has a key other than the allowed ones, or
  has a value of the wrong type; any table, block or LayerNorm has the wrong shape; any tensor's
  dtype differs from `tok_emb`; `tok_emb` is not floating-point.
- Must not mutate `idx` or any tensor in `params`.
- Differentiable: the result stays on the autograd graph. When an input has
  `requires_grad=True`, backpropagating through the result gives the gradient of the formula
  above. Do not detach, and do not leave PyTorch (`.item()`, `.numpy()`, `torch.no_grad()`) on
  the way.

## Examples

```python
>>> tok_emb = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float64)  # V=3, D=2
>>> params = {"tok_emb": tok_emb, "pos_emb": torch.zeros(1, 2, dtype=torch.float64),
...           "blocks": [], "ln_f_gamma": torch.ones(2, dtype=torch.float64),
...           "ln_f_beta": torch.zeros(2, dtype=torch.float64)}   # no lm_head: tied
>>> gpt_forward(torch.tensor([[1]]), params, num_heads=1).shape
torch.Size([1, 1, 3])
>>> gpt_forward(torch.tensor([[1]]), params, num_heads=1)
tensor([[[-1.0000,  1.0000,  0.0000]]], dtype=torch.float64)
# Token 1 embeds to [0, 1]. LayerNorm gives about [-1, 1]. Tied logits are h @ tok_emb.T,
# so the scores are [-1, 1, 0]. The small epsilon makes the first two entries -0.99998 and 0.99998.
```

```python
>>> # Same params, with one block whose sub-layers are zero (all projections and biases 0,
>>> # both LayerNorm gammas 1, betas 0). A zero block passes the stream through unchanged,
>>> # so the logits equal the logits above.
>>> gpt_forward(torch.tensor([[1]]), params_with_one_zero_block, num_heads=1)
tensor([[[-1.0000,  1.0000,  0.0000]]], dtype=torch.float64)
```

```python
>>> gpt_forward(torch.tensor([[3]]), params, num_heads=1)
ValueError: token id 3 is outside the vocabulary of size 3
```

## Constraints

- Allowed: PyTorch tensor operations such as indexing, `matmul`, `reshape`, `transpose`, `softmax`,
  `masked_fill`, `triu`, `erf`, `sqrt`, `torch.is_tensor`, plus the standard library (`math`).
- Not allowed: `torch.nn`, including `torch.nn.functional`, `torch.nn.Embedding`,
  `torch.nn.LayerNorm`, `torch.nn.Linear`, `torch.nn.MultiheadAttention` and
  `torch.nn.TransformerEncoderLayer`. Those are the test oracles. Also not allowed: the top-level
  shortcuts `torch.layer_norm` and `torch.embedding`, which do the LayerNorm and the lookup in one
  call. Write the embedding lookup, LayerNorm, attention, GELU and the head yourself.

## Done looks like

All public tests pass. Then **Submit** runs the hidden tests. They check: agreement with an
independent reference on seeded batches, for tied and untied heads, several shapes and zero, one
and two blocks; the tied head sharing `tok_emb` (the output must equal the output with
`lm_head = tok_emb.T` passed explicitly); no information from future tokens reaching earlier
logits; the position table being added per position, not per batch row; batch rows processed
independently; the context boundary `T = C` accepted and `T = C + 1` rejected; out-of-range and
negative token ids rejected; large and small magnitudes; float32 inputs returning float32 logits;
inputs and parameters left unchanged; and every `ValueError` rule above. A wrong variant gets a
named message, for example an untied head where the tie was intended, a transposed head or a
missing position table.

The hidden tests also backpropagate through the result and compare the gradients with PyTorch's
own.

The hidden tests draw their bulk inputs from the real datasets `breast_cancer`,
`gutenberg_frankenstein`, `gutenberg_pride_prejudice` and `gutenberg_walden` (see
`docs/DATASETS.md`).

# Position-wise GELU feed-forward block

Implement `gelu_ffn(X, W1, b1, W2, b2)`. It is the two-layer feed-forward block of a Transformer.
It applies the same network to every position independently: expand the features from `d_model`
to `d_ff` with a linear layer, apply the exact GELU, then project back to `d_model` with a second
linear layer. Positions do not interact here; attention is a separate block.

For one position with features `x` (length `d_model`):

`FFN(x) = W2 @ GELU(W1 @ x + b1) + b2`, where `GELU(z) = z * Phi(z)` and
`Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))` is the standard normal CDF. This is the exact GELU, not the
tanh approximation.

## Contract

- `X`: `torch.float64` tensor of shape `(..., d_model)`, with at least one dimension. The leading
  axes (for example batch and sequence) are arbitrary and are treated as independent positions.
- `W1`: `torch.float64`, shape `(d_ff, d_model)`, in PyTorch `nn.Linear` layout. The expansion
  is `z = x @ W1.T + b1`.
- `b1`: shape `(d_ff,)`.
- `W2`: shape `(d_model, d_ff)`, the projection back, also in `nn.Linear` layout.
- `b2`: shape `(d_model,)`.
- `d_ff` may be 0. Then the hidden layer is empty and the output is `b2` at every position.
- Returns: `torch.float64` tensor with the same shape as `X`, i.e. `(..., d_model)`.
- Raises `ValueError` when: `X` is a 0-dimensional tensor; the last dimension of `X` is not
  `W1.shape[1]`; `W1` is not 2-D; `W2` is not `(d_model, d_ff)` with `d_model = W1.shape[1]` and
  `d_ff = W1.shape[0]`; `b1` is not `(d_ff,)`; `b2` is not `(d_model,)`.
- Must not mutate inputs.

## Examples

```python
>>> X = torch.tensor([[1.0, 0.0]], dtype=torch.float64)         # one position, d_model = 2
>>> W1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)  # d_ff = 2
>>> b1 = torch.zeros(2, dtype=torch.float64)
>>> W2 = torch.eye(2, dtype=torch.float64)
>>> b2 = torch.zeros(2, dtype=torch.float64)
>>> gelu_ffn(X, W1, b1, W2, b2)
tensor([[0.8413, 0.0000]], dtype=torch.float64)   # GELU(1) = Phi(1) ~ 0.8413 and GELU(0) = 0
```

```python
>>> gelu_ffn(X, W1, b1, W2[:, :1], b2)   # W2 has the wrong d_ff
ValueError: W2 must have shape (d_model, d_ff)
```

## Constraints

- Allowed: the `torch` package, for example `torch.matmul`, `torch.erf`, `torch.special`,
  `torch.sqrt` and `torch.as_tensor`, and the `math` module.
- Not allowed: `torch.nn` in any form, including `torch.nn.functional.gelu`,
  `torch.nn.functional.linear` and `torch.nn.Linear`. They are the test oracles, not the tool.
  Write the two linear maps and the GELU yourself.

## Done looks like

All public tests pass. Then **Submit** runs the hidden tests. They cover: agreement with a
PyTorch reference on seeded random batches of several shapes; the exact erf form of GELU, not the
tanh approximation (checked where the two differ); each position computed independently of the
others (changing one token must not change another); the empty hidden layer (`d_ff = 0`); the bias
and weight orientation on non-square, non-symmetric weights; the `ValueError` rules (scalar input,
feature mismatch, weight and bias shapes); large and small magnitudes, including negative tails
where `1 + erf` cancels; float64 dtype and shape; and inputs left unchanged.

The hidden tests draw their bulk inputs from the real dataset `breast_cancer` (see
`docs/DATASETS.md`).

# Gated SwiGLU feed-forward block

Implement `swiglu_ffn(X, W, V, W2)`. It is the gated variant of the Transformer feed-forward block
(Shazeer 2020). Two linear maps read the same input. One is passed through the SiLU (Swish)
nonlinearity and acts as a gate. The other is left linear. Their elementwise product goes through a
third linear map back to `d_model`. Like the plain block, the same network is applied to every
position independently, and the layer has no biases.

For one position with features `x` (length `d_model`):

`SwiGLU(x) = W2 @ ( SiLU(W @ x) * (V @ x) )`, where `SiLU(z) = z * sigmoid(z)`.

The `*` is elementwise. `W` is the gate projection, `V` is the up projection, and `W2` is the down
projection.

## Contract

- `X`: `torch.float64` tensor of shape `(..., d_model)`, with at least one dimension. Leading axes
  are independent positions.
- `W`: `torch.float64`, shape `(d_ff, d_model)`, in PyTorch `nn.Linear` layout. The gate is
  `g = X @ W.T`.
- `V`: shape `(d_ff, d_model)`, same layout. The up projection is `u = X @ V.T`.
- `W2`: shape `(d_model, d_ff)`, same layout. The output is `(SiLU(g) * u) @ W2.T`.
- `d_ff` may be 0. Then the output is zeros of shape `(..., d_model)`.
- Returns: `torch.float64` tensor of shape `(..., d_model)`.
- Raises `ValueError` when: `X` is 0-dimensional; `W` or `V` is not 2-D; `W` and `V` have different
  shapes; the last dimension of `X` is not `W.shape[1]`; `W2` is not `(d_model, d_ff)`.
- Must not mutate inputs.
- Differentiable: the result stays on the autograd graph. When an input has
  `requires_grad=True`, backpropagating through the result gives the gradient of the formula
  above. Do not detach, and do not leave PyTorch (`.item()`, `.numpy()`, `torch.no_grad()`) on
  the way.

## Examples

```python
>>> X = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
>>> W  = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
>>> V  = torch.tensor([[1.0, 1.0], [0.0, 1.0]], dtype=torch.float64)
>>> W2 = torch.eye(2, dtype=torch.float64)
>>> swiglu_ffn(X, W, V, W2)
tensor([[2.1932, 3.5232]], dtype=torch.float64)
# gate = [1, 2], up = [3, 2]; SiLU(1) = 0.7311, SiLU(2) = 1.7616; products 2.1932 and 3.5232.
```

```python
>>> swiglu_ffn(torch.zeros(1, 2, dtype=torch.float64), W, V, W2)
tensor([[0., 0.]], dtype=torch.float64)   # a zero gate input gives SiLU(0) = 0
```

## Constraints

- Allowed: the `torch` package, for example `torch.matmul`, `torch.sigmoid`, `torch.special` and
  `torch.clamp`, and the `math` module.
- Not allowed: `torch.nn` in any form, including `torch.nn.functional.silu`, and
  `torch.nn.functional.linear`. They are the test oracles, not the tool. Write the three linear maps,
  the SiLU and the product yourself.

## Done looks like

All public tests pass. Then **Submit** runs the hidden tests. They cover: agreement with a
PyTorch reference on seeded random batches; that the SiLU is applied to the gate path and not to the
up path (swapping the roles changes the output); the SiLU itself rather than ReLU or GELU; the
`nn.Linear` orientation on non-square weights; the bias-free contract; positions computed
independently; a zero input giving a zero output; the empty gate (`d_ff = 0`); large and small
magnitudes; the `ValueError` rules (scalar input, mismatched gate and up shapes, wrong down shape,
feature mismatch); float64 dtype and shape; and inputs left unchanged.

The hidden tests also backpropagate through the result and compare the gradients with PyTorch's
own.

The hidden tests draw their bulk inputs from the real dataset `breast_cancer` (see
`docs/DATASETS.md`).

# LoRA linear layer: forward pass and merge

Implement `lora_linear(x, W, b, A, B, alpha, merge=False)`. It computes the output of one linear layer
that has a frozen base weight `W` and a trainable low-rank adapter `(A, B)`:

$$
y = x W^\top + b + \frac{\alpha}{r}\, x A^\top B^\top, \qquad r = \text{rank of the adapter}.
$$

Symbols: `x` has shape `(..., d_in)`; `W` is the base weight `(d_out, d_in)`; `b` is the bias `(d_out,)`;
`A` is the down-projection `(r, d_in)`; `B` is the up-projection `(d_out, r)`; `alpha` is a scale.

With `merge=False`, the function computes the two products as written. With `merge=True`, it first folds
the update into one weight, $W' = W + \frac{\alpha}{r} B A$ with shape `(d_out, d_in)`, and computes
$y = x W'^\top + b$. Both paths return the same values.

The base `W` and `b` are frozen by the caller: they do not require gradients. `A` and `B` do, and
gradients must flow through both paths into `A` and `B`.

## Contract

- `x`: floating tensor, shape `(..., d_in)` with at least one dimension. Leading dimensions are free.
- `W`: floating tensor, shape `(d_out, d_in)`, PyTorch `nn.Linear` layout (a projection is `x @ W.T`).
- `b`: floating tensor of shape `(d_out,)`, or `None` for no bias.
- `A`: floating tensor, shape `(r, d_in)`, with `r >= 1`.
- `B`: floating tensor, shape `(d_out, r)`.
- `alpha`: a finite real number (`int` or `float`, not `bool`). The scale is `alpha / r`, with `r` taken
  from `A.shape[0]`.
- `merge`: `bool`. `False` uses the two thin products. `True` builds `W'` first.
- `x`, `W`, `A`, `B` and `b` (when given) must share one floating dtype.
- Returns: a tensor of shape `(..., d_out)` with the dtype of `x`.
- Raises `ValueError` when: `x` has no dimension; `W` is not 2-D; `W.shape[1] != x.shape[-1]`; `A` is not
  2-D or `A.shape[1] != d_in`; `A` has zero rows; `B` is not 2-D or `B.shape != (d_out, r)`; `b` is not
  `None` and its shape is not `(d_out,)`; `alpha` is not a finite real number; the tensors do not share
  one floating dtype.
- Must not mutate any input. In particular, `merge=True` builds a new `W'`; it never writes into `W`.
- Must keep the autograd graph of `A` and `B`. Do not call `.detach()` or `.data` on them.

## Examples

```python
>>> x = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
>>> W = torch.eye(2, dtype=torch.float64)                         # base layer is the identity
>>> A = torch.tensor([[1.0, 0.0]], dtype=torch.float64)           # r = 1, d_in = 2
>>> B = torch.tensor([[0.0], [1.0]], dtype=torch.float64)         # d_out = 2, r = 1
>>> lora_linear(x, W, None, A, B, alpha=2.0)                      # scale = alpha / r = 2
tensor([[1., 4.]], dtype=torch.float64)                           # base [1, 2] + 2 * [0, 1]
>>> lora_linear(x, W, None, A, B, alpha=2.0, merge=True)          # W' = [[1, 0], [2, 1]]
tensor([[1., 4.]], dtype=torch.float64)
```

```python
>>> B0 = torch.zeros(2, 1, dtype=torch.float64)                   # zero-initialised B
>>> lora_linear(x, W, None, A, B0, alpha=2.0)                     # training starts at the base model
tensor([[1., 2.]], dtype=torch.float64)
```

```python
>>> lora_linear(x, W, None, A, torch.zeros(3, 1, dtype=torch.float64), alpha=2.0)
ValueError: B must have shape (d_out, r) = (2, 1)     # B has 3 rows, but d_out is 2
```

## Constraints

- Allowed: the `torch` package for tensor math, for example `@`, `matmul`, `.T`, `transpose`, `zeros`,
  `is_floating_point` and `math.isfinite`.
- Not allowed: `torch.nn` in any form, including `torch.nn.Linear`, `torch.nn.functional.linear`, and
  `torch.nn.Parameter`. The base layer is a plain matrix product you write yourself. The `peft` package is
  not available.

## Done looks like

All public tests pass. **Submit** then runs the hidden tests. They compare against a reference built from
`torch.nn.functional.linear` on seeded random inputs, with 3-D inputs and several ranks, including a rank
of one and a rank equal to `d_in`. They check that the merged and unmerged outputs agree, that a zero `B`
returns the base output, that the scale is `alpha / r` rather than `alpha` or `1 / r`, and that the bias is
added once. They check the gradient of a random linear loss with respect to `A` and `B` against a float64
finite-difference check, on both paths. They check that `W` and `b` receive no gradient, that no input
changes, that a large `alpha` with tiny factors and a small `alpha` with large factors both keep their
precision, the output dtype, and every `ValueError` rule.

The hidden tests draw their bulk inputs from the real dataset `breast_cancer` (see
`docs/DATASETS.md`).

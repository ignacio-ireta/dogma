# Straight-through quantizer

Implement `ste_quantize(x, step)`. The forward pass snaps every entry of `x` to the nearest multiple
of `step`. The backward pass pretends the quantizer was the identity: the gradient reaching `x` is
the gradient arriving at the output, unchanged. This is the straight-through estimator used to
train quantized or binarized networks, where `round` has zero gradient almost everywhere and
would otherwise stop learning.

## Contract

- `x`: floating tensor of any shape, including 0-dimensional. It may or may not have
  `requires_grad=True`.
- `step`: a finite Python `float` or `int` greater than zero. Booleans are invalid.
- Forward: `y = torch.round(x / step) * step`. `torch.round` rounds half to even, so
  `0.5 -> 0`, `1.5 -> 2`, `2.5 -> 2`. The values match that formula up to floating-point rounding, because
  the straight-through form computes `x + (quantized - x)`. The result has the shape and dtype of `x`.
- Backward: with respect to `x`, the gradient is the incoming gradient unchanged, everywhere. There
  is no clipping, so values far outside any range still pass the gradient through.
- `y` requires grad exactly when `x` does. If `x` does not require grad, `y` must not either.
- Must not mutate `x`.
- Raises: `ValueError` when `x` is not floating, or when `step` is not a finite number greater
  than zero.

## Examples

```python
ste_quantize(torch.tensor([0.4, 0.6, 1.3]), 0.5)
# forward values -> tensor([0.5, 0.5, 1.5])
```

```python
x = torch.tensor([0.2, 1.9], requires_grad=True)
y = ste_quantize(x, 1.0)  # y -> tensor([0., 2.])
(3 * y).sum().backward()
# x.grad -> tensor([3., 3.])   (identity, not zero: the round itself has zero slope)
```

## Constraints

- Allowed: `detach`, `torch.round`, arithmetic, and the straight-through construction with
  `detach` (a sum of a differentiable term and a detached difference is the usual form).
- Not allowed: `torch.autograd.Function` (custom backward), `torch.autograd.functional`,
  `torch.func`.

## Done looks like

Your forward matches round-half-to-even quantization. Your backward is the identity for every
entry, including entries outside a clipping range. The hidden tests check the identity gradient
through a chain of operations, a float64 finite-difference check over one full step, tie handling,
dtype and shape, invalid steps and inputs, and that `x` is not mutated.

The hidden tests draw their bulk inputs from the real dataset `breast_cancer` (see
`docs/DATASETS.md`).

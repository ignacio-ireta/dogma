# Gradient at a point without side effects

Implement `grad_at(f, x)`. It returns the gradient of a scalar function `f` at the point `x`, using
`torch.autograd.grad`. The caller's tensor and any `.grad` field must be left exactly as they were.
This is the pattern behind gradient checks, sensitivity analysis and attribution: you ask for a
derivative and expect no hidden state to change.

## Contract

- `f`: a callable that takes a 1-D float tensor of shape `(n,)` and returns a 0-dimensional tensor.
- `x`: 1-D floating tensor of shape `(n,)`, `n >= 1`. It may or may not have `requires_grad=True`,
  and it may already carry a `.grad` value.
- Returns: a tensor of shape `(n,)` and the same dtype as `x`, equal to the derivative of `f` with
  respect to `x` at the point `x`. The result is detached: `requires_grad` is `False` and it has no
  `grad_fn`.
- If `f`'s output does not depend on `x` (a constant, or no path from `x`), the gradient is all zeros.
- Must work when called inside `torch.no_grad()`.
- Must leave `x` unchanged: its values, its `requires_grad` flag and its `.grad` field.
- Raises: `ValueError` when `x` is not 1-D or not floating, or when `f(x)` is not 0-dimensional.

## Examples

```python
grad_at(lambda v: (v**2).sum(), torch.tensor([1.0, 2.0, 3.0]))
# -> tensor([2., 4., 6.])
```

```python
x = torch.tensor([1.0, 2.0], requires_grad=True)
g = grad_at(lambda v: (v**3).sum(), x)
# g -> tensor([ 3., 12.]);  x.requires_grad -> True;  x.grad -> None
```

## Constraints

- Allowed: `torch.autograd.grad`, `torch.enable_grad`, `detach`, `clone`, `requires_grad_` on a
  tensor you create yourself.
- Not allowed: `torch.Tensor.backward` (which also covers `torch.autograd.backward`),
  `torch.autograd.functional` and `torch.func`. Those bypass the point of the exercise.
- Do not write `.grad` on any tensor the caller passed in.

## Done looks like

Your function returns the analytic gradient for smooth functions, agrees with central finite
differences in float64, and leaves the caller's tensor state unchanged. The hidden tests check
accumulated and preexisting `.grad` fields, `no_grad` contexts, constant and unused-input functions,
float32 dtype, large and small magnitudes, and invalid shapes, dtypes and output ranks.

The hidden tests draw their bulk inputs from the real dataset `breast_cancer` (see
`docs/DATASETS.md`).

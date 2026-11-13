# Layer normalization forward over the last dimension

Implement `layer_norm(x, weight, bias, eps=1e-5)`. Normalize every length-`D` vector along the last
dimension of `x`, separately for each example and each leading position. Apply the per-feature gain
and bias, and return a tensor of the same shape as `x`. Keep no state between calls.

## Contract

- `x`: floating-point tensor with `ndim >= 1`, shape `(..., D)`, `D >= 1`. Leading dimensions may be any size, including zero. Must not be modified.
- `weight`, `bias`: tensors of shape `(D,)`. Must not be modified.
- `eps`: float `> 0`, added to the variance inside the square root.
- Statistics: the mean and the **biased** variance (divide by `D`) over the last dimension, with the mean subtracted before squaring.
- Returns: a tensor with the shape and dtype of `x`, equal to `weight * (x - mean) / sqrt(var + eps) + bias`.
- Raises `ValueError` when: `x` is a 0-dimensional or non-floating tensor; `D == 0`; `weight` or `bias` does not have shape `(D,)`; `eps <= 0`.
- Must not mutate inputs.
- Differentiable: the result stays on the autograd graph. When an input has
  `requires_grad=True`, backpropagating through the result gives the gradient of the formula
  above. Do not detach, and do not leave PyTorch (`.item()`, `.numpy()`, `torch.no_grad()`) on
  the way.

## Examples

The examples write tensors as nested lists for brevity. The arguments are `torch` tensors, for example `torch.tensor([[1.0, 2.0, 3.0]])`.

```python
x = [[1.0, 2.0, 3.0]]  # one example, D=3
weight, bias = [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]
layer_norm(x, weight, bias, eps=1e-12)
# ~= [[-1.2247, 0.0, 1.2247]]   (mean 2, biased variance 2/3)
```

```python
x = [
    [1.0, 2.0, 3.0],  # two examples, D=3
    [10.0, 10.0, 10.0],
]  # a constant example
weight, bias = [2.0, 1.0, 0.5], [0.0, 0.0, 1.0]
layer_norm(x, weight, bias, eps=1e-12)
# ~= [[-2.4495, 0.0, 1.6124],
#     [ 0.0,    0.0, 1.0   ]]  (constant row: the centred values are zero, so the output is the bias)
```

## Constraints

- Allowed: Python standard library and PyTorch tensor operations, such as `mean`, `sqrt`, `clone`, and broadcasting.
- Not allowed: `torch.nn` (this covers `torch.nn.functional`, `nn` modules and `from torch import nn`), `torch.ops`, and any library routine that normalizes the last dimension as a whole: `torch.layer_norm`, `torch.native_layer_norm`, `torch.group_norm` (a single group normalizes over all features), and `torch.instance_norm`. Write the statistics yourself.

## Done looks like

The examples and the single-feature edge case pass. The hidden tests compare outputs with PyTorch's layer norm over random inputs with 1 to 4 leading dimensions, and check that each example is normalized independently of the others, that the statistics run over the last axis, that the mean is subtracted before scaling, that eps sits inside the square root, that the variance is biased, and that large common offsets and tiny magnitudes are handled. They also cover float32 dtype, empty leading dimensions, invalid arguments, and no mutation.

The hidden tests also backpropagate through the result and compare the gradients with PyTorch's
own.

The hidden tests draw their bulk inputs from the real dataset `breast_cancer` (see
`docs/DATASETS.md`).

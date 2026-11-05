# RMSNorm forward with gain and eps

Implement `rms_norm(x, gain, eps=1e-6)`. Divide every length-`D` vector along the last dimension of
`x` by its root mean square, multiply by a per-feature gain, and return the result. Do **not**
subtract the mean. Keep no state between calls.

## Contract

- `x`: floating-point tensor with `ndim >= 1`, shape `(..., D)`, `D >= 1`. Leading dimensions may be any size, including zero. Must not be modified.
- `gain`: tensor of shape `(D,)`. Must not be modified.
- `eps`: float `> 0`. Added to the mean square **inside** the square root.
- Statistic: `mean(x**2)` over the last dimension, with `keepdim`.
- Returns: a tensor with the shape and dtype of `x`, equal to `gain * x / sqrt(mean(x**2) + eps)`.
- Raises `ValueError` when: `x` is 0-dimensional or non-floating; `D == 0`; `gain` does not have shape `(D,)`; `eps <= 0`.
- Must not mutate inputs.

## Examples

The examples write tensors as nested lists for brevity. The arguments are `torch` tensors, for example `torch.tensor([[1.0, 3.0]])`.

```python
x = [[1.0, 3.0]]  # mean square 5, root mean square sqrt(5)
gain = [1.0, 1.0]
rms_norm(x, gain, eps=1e-12)
# ~= [[0.4472, 1.3416]]
```

```python
x = [[3.0, 4.0]]  # mean square 12.5
gain = [2.0, 0.5]
rms_norm(x, gain, eps=1e-12)
# ~= [[1.6971, 0.5657]]
```

```python
x = [[0.0, 0.0]]  # zero input: the output is zero, not NaN
rms_norm(x, [1.0, 1.0], eps=1e-6)
# -> [[0.0, 0.0]]
```

## Constraints

- Allowed: Python standard library and PyTorch tensor operations, such as `mean`, `sqrt`, broadcasting.
- Not allowed: `torch.nn` (this covers `torch.nn.functional`, `nn` modules and `from torch import nn`), `torch.ops`, `torch.rms_norm`, and `torch._fused_rms_norm`. Write the statistic yourself.

## Done looks like

The examples, the zero-input case and the single-feature case pass. The hidden tests compare outputs with PyTorch's RMSNorm over random inputs with 1 to 4 leading dimensions, and check that the mean is not subtracted (an additive offset must change the output), that eps sits inside the square root (small magnitudes expose outside placement), that the output has unit mean square before the gain, that the statistic runs over the last axis, that examples are independent, that very large and very small magnitudes are handled, and that float32, empty leading dimensions, invalid arguments and in-place mutation are covered.

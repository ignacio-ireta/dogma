# Batch normalization backward (training mode)

Implement `batch_norm_backward(grad_y, x, gamma, eps=1e-5)` for the training-mode forward pass of
batch normalization on an `(N, C)` batch. Return the gradients of the scalar objective
`L = sum(grad_y * y)` with respect to `x`, `gamma` and `beta`. Use the closed form. Do not call
autograd.

The forward pass being differentiated is:

- `mu = mean over rows of x`, shape `(C,)`
- `var = biased variance over rows of x` (divide by `N`), shape `(C,)`
- `x_hat = (x - mu) / sqrt(var + eps)`, shape `(N, C)`
- `y = gamma * x_hat + beta`

## Contract

- `grad_y`: floating-point tensor, shape `(N, C)`. Upstream gradient `dL/dy`. Must not be modified.
- `x`: floating-point tensor, shape `(N, C)`, with `N >= 2` and `C >= 1`. Must not be modified.
- `gamma`: tensor of shape `(C,)`. Must not be modified.
- `eps`: float `> 0`.
- Returns: `(grad_x, grad_gamma, grad_beta)`, with shapes `(N, C)`, `(C,)`, `(C,)`, and the dtype of `x`.
  - `grad_beta[c] = sum over rows of grad_y[:, c]`
  - `grad_gamma[c] = sum over rows of grad_y[:, c] * x_hat[:, c]`
  - `grad_x` is the gradient through the batch statistics, so it includes the mean and variance terms.
- Raises `ValueError` when: `x` or `grad_y` is not 2-D or not floating-point; `grad_y` and `x` have different shapes; `N < 2`; `gamma` does not have shape `(C,)`; `eps <= 0`.

## Examples

The examples write tensors as nested lists for brevity. The arguments are `torch` tensors, for example `torch.tensor([[1.0], [3.0]])`.

```python
x = [[1.0], [3.0]]  # N=2, C=1: x_hat is [-1, 1] whatever the scale
grad_y = [[1.0], [0.0]]
gamma = [2.0]
batch_norm_backward(grad_y, x, gamma, eps=1e-12)
# grad_x ~= [[0.0], [0.0]]          (two rows: the normalized values never move, so dx vanishes)
# grad_gamma ~= [-1.0], grad_beta = [1.0]
```

```python
x = [[1.0], [3.0], [5.0]]  # N=3, C=1
grad_y = [[1.0], [1.0], [1.0]]  # constant upstream gradient
gamma = [2.0]
batch_norm_backward(grad_y, x, gamma, eps=1e-12)
# grad_x = [[0.0], [0.0], [0.0]]
# grad_gamma = [0.0], grad_beta = [3.0]
```

## Constraints

- Allowed: Python standard library and PyTorch tensor operations, such as `mean`, `sum`, `sqrt`, `clone`.
- Not allowed: `torch.nn` (this covers `torch.nn.functional` and `nn` modules), `torch.autograd`, `torch.func`, `torch.ops`, and any library routine that computes the batch-norm forward or backward pass as a whole: `torch.batch_norm`, `torch.native_batch_norm`, `torch._native_batch_norm_legit`, `torch._native_batch_norm_legit_no_training`, `torch._batch_norm_impl_index`, `torch.batch_norm_backward_elemt`, `torch.batch_norm_backward_reduce`, `torch.batch_norm_stats`, `torch.batch_norm_elemt`, `torch.batch_norm_update_stats`, `torch.batch_norm_gather_stats`, `torch.cudnn_batch_norm`, `torch.miopen_batch_norm`, `torch.instance_norm`, `torch.group_norm`. Derive the gradients in closed form and do not call `backward()`.

## Done looks like

The examples pass, and the single-row and shape errors raise `ValueError`. The hidden tests compare
`grad_x`, `grad_gamma` and `grad_beta` with autograd through PyTorch's batch norm on random batches,
check the same gradients against central finite differences in float64, and cover a constant
upstream gradient, an upstream gradient that is not constant across rows, several shapes where rows
and features differ, large common offsets, invalid arguments, and no mutation of inputs.

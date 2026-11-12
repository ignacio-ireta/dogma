# One SGD step with momentum

Implement `sgd_momentum_step(params, grads, velocity, lr, momentum)`. It takes one optimizer
step over a list of parameter tensors, using a velocity buffer per tensor. Use the PyTorch
convention: the buffer is an exponential moving sum of gradients, and the learning rate
multiplies the buffer only in the parameter update. Return the new parameters and the new
buffers, so the caller can pass the buffers back in on the next step.

Use `velocity=None` on the first step. On that step the buffer is the raw gradient.

## Contract

- `params`, `grads`: lists of floating-point tensors of equal length. `grads[i]` has the same
  shape and dtype as `params[i]`.
- `velocity`: `None` before the first step, otherwise a list with one tensor per parameter,
  each the shape of its parameter.
- `lr`: finite and `>= 0`. `momentum`: finite and in `[0, 1)`.
- Per tensor, with gradient `g` and buffer `v`:
  - first step (`velocity is None`): `v = g`
  - later steps: `v = momentum * v_prev + g`
  - update: `p_new = p - lr * v`
- Returns: `(new_params, new_velocity)`. Both are lists of tensors with the input shapes and
  dtypes. `new_velocity` holds the unscaled buffer `v`, not `lr * v`.
- Raises `ValueError` when lengths differ, a gradient or buffer shape differs from its
  parameter, `lr` or `momentum` is out of range, or `lr` or `momentum` is not finite.
- Must not mutate inputs. Return new tensors.

## Examples

```python
params, velocity = sgd_momentum_step(
    [torch.tensor([1.0, 2.0])], [torch.tensor([0.5, -1.0])], None, lr=0.1, momentum=0.9
)
# params   -> [tensor([0.9500, 2.1000])]
# velocity -> [tensor([ 0.5000, -1.0000])]
```

Feed the same gradient again with `velocity` from the first call:

```python
params, velocity = sgd_momentum_step(params, [torch.tensor([0.5, -1.0])], velocity, 0.1, 0.9)
# velocity -> [tensor([ 0.9500, -1.9000])]   (0.9 * [0.5, -1.0] + [0.5, -1.0])
# params   -> [tensor([0.8550, 2.2900])]
```

## Constraints

Allowed: Python, `torch` tensor operations, and `math`. Not allowed: `torch.optim`, which
contains the optimizer you are checking against. Write the update yourself.

## Done looks like

The public tests pass: a first step, a second step that uses the buffer, and a shape or length
error. The hidden tests compare your trajectory with `torch.optim.SGD` over several steps on
seeded random tensors. They also check the first-step rule, the momentum-zero case, the
buffer convention in the returned state, input immutability, invalid arguments, scale
extremes and dtype. A failing test names the mistake, such as a damped first step or a
learning rate that leaked into the buffer.

The hidden tests draw their bulk inputs from the real dataset `breast_cancer` (see
`docs/DATASETS.md`).

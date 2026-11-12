# Nesterov momentum with weight decay

Implement `sgd_nesterov_step(params, grads, velocity, *, lr, momentum, weight_decay)`. The `lr`,
`momentum` and `weight_decay` arguments are keyword-only. It is
the same buffer-based update as the momentum step, with two changes. First, L2 weight decay
is coupled: the decay term joins the gradient before momentum sees it. Second, the update
uses the Nesterov look-ahead: it takes the gradient one momentum move ahead along the
buffer. This matches `torch.optim.SGD(..., nesterov=True, weight_decay=...)`.

## Contract

- `params`, `grads`: lists of floating-point tensors of equal length. `grads[i]` has the same
  shape and dtype as `params[i]`.
- `velocity`: `None` before the first step, otherwise one buffer per parameter, each the shape
  of its parameter.
- `lr`, `weight_decay`: finite and `>= 0`. `momentum`: finite and in `(0, 1)`.
- Per tensor, with gradient `g`, parameter `p`, and buffer `v`:
  - decayed gradient: `d = g + weight_decay * p`
  - first step (`velocity is None`): `v = d`; later steps: `v = momentum * v_prev + d`
  - look-ahead: `d_nes = d + momentum * v`
  - update: `p_new = p - lr * d_nes`
- Returns `(new_params, new_velocity)`, with the input shapes and dtypes. `new_velocity` holds
  the unscaled buffer `v`, built from gradient plus decay.
- Raises `ValueError` for mismatched lengths or shapes, out-of-range or non-finite scalars
  (including `momentum = 0`, which is not Nesterov here).
- Must not mutate inputs.

## Examples

```python
params, velocity = sgd_nesterov_step(
    [torch.tensor([1.0])], [torch.tensor([0.5])], None, lr=0.1, momentum=0.9, weight_decay=0.1
)
# d = 0.5 + 0.1 * 1.0 = 0.6, v = 0.6, look-ahead = 0.6 + 0.9 * 0.6 = 1.14
# params   -> [tensor([0.8860])]
# velocity -> [tensor([0.6000])]
```

Second step, with the same gradient and the buffer from the first call:

```python
params, velocity = sgd_nesterov_step(
    params, [torch.tensor([0.5])], velocity, lr=0.1, momentum=0.9, weight_decay=0.1
)
# velocity -> [tensor([1.1286])]
# params   -> [tensor([0.7256])]
```

## Constraints

Allowed: Python, `torch` tensor operations, and `math`. Not allowed: `torch.optim`, which
contains the optimizer you are checking against. Write the update yourself.

## Done looks like

The public tests pass for two steps and for an invalid weight decay. Hidden tests compare
your trajectory with `torch.optim.SGD(nesterov=True)` on seeded random tensors. They also
check where decay enters the update, the zero-decay case, the buffer convention, input
immutability, invalid arguments, scale extremes and dtype. A failing test names the kind of
mistake, such as decay added after momentum or a classical update in place of the look-ahead.

The hidden tests draw their bulk inputs from the real dataset `breast_cancer` (see
`docs/DATASETS.md`).

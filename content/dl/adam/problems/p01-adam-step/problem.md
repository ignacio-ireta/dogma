# One Adam step with bias correction

Implement `adam_step(params, grads, state, *, lr, beta1=0.9, beta2=0.999, eps=1e-8)`. It takes
one Adam update over a list of parameter tensors. Each tensor carries a first moment `m`
(an average of gradients) and a second moment `v` (an average of squared gradients). A step
counter `t` lets the update correct the bias that a zero start introduces. Return the new
parameters and the new state, so the caller can pass the state back in on the next update.

Use `state=None` before the first update. On that call the moments start at zero and `t` is 0.

## Contract

- `params`, `grads`: lists of floating-point tensors of equal length. `grads[i]` has the same
  shape and dtype as `params[i]`.
- `state`: `None`, or the tuple `(m, v, t)`. `m` and `v` are lists with one tensor per
  parameter, each the shape of its parameter. `t` is a non-negative `int` (not `bool`): the
  number of updates already taken.
- `lr`: finite and `>= 0`. `beta1`, `beta2`: in `[0, 1)`. `eps`: finite and `> 0`.
- With `t_new = t + 1`, per tensor, using gradient `g`:
  - `m = beta1 * m + (1 - beta1) * g`
  - `v = beta2 * v + (1 - beta2) * g * g`
  - `m_hat = m / (1 - beta1**t_new)`, `v_hat = v / (1 - beta2**t_new)`
  - `p_new = p - lr * m_hat / (sqrt(v_hat) + eps)`
- Returns `(new_params, (new_m, new_v, t_new))`, with the input shapes and dtypes.
- Raises `ValueError` when lengths or shapes differ, when a scalar is out of range or not finite,
  or when `state` is malformed (wrong list length, wrong moment shape, or a non-integer or boolean
  `t`).
- Must not mutate inputs.

## Examples

```python
params, state = adam_step([torch.tensor([1.0])], [torch.tensor([2.0])], None, lr=0.1)
# t_new = 1: m_hat = 2, v_hat = 4, step = 0.1 * 2 / (2 + eps)
# params -> [tensor([0.9000])]
```

Second update with the same gradient:

```python
params, state = adam_step(params, [torch.tensor([2.0])], state, lr=0.1)
# params -> [tensor([0.8000])]   (each update moves about lr for a constant gradient)
```

## Constraints

Allowed: Python, `torch` tensor operations, and `math`. Not allowed: `torch.optim`, which
contains the optimizer you are checking against. Write the update yourself.

## Done looks like

The public tests pass for a first step at two gradient scales, a constant-gradient second step
and an invalid beta. Hidden tests compare your trajectory with `torch.optim.Adam` over up to 200
steps on seeded random tensors. They also check the first step against the bias-corrected formula,
the step counter, the placement of eps, zero gradients, input immutability, malformed state and
arguments, scale extremes and dtype. A failing test names the mistake, such as a missing bias
correction, an eps inside the root, or a counter that starts at zero.

The hidden tests draw their bulk inputs from the real dataset `breast_cancer` (see
`docs/DATASETS.md`).

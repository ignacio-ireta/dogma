# One AdamW step with decoupled weight decay

Implement `adamw_step(params, grads, state, *, lr, beta1=0.9, beta2=0.999, eps=1e-8,
weight_decay=0.01)`. It is one Adam update with decoupled weight decay: before the adaptive step,
each weight is shrunk by the factor `1 - lr * weight_decay`. The decay never enters the gradient, so
the adaptive second moment never sees it. This matches `torch.optim.AdamW`.

## Contract

- `params`, `grads`: lists of floating-point tensors of equal length. `grads[i]` has the same
  shape and dtype as `params[i]`.
- `state`: `None`, or the tuple `(m, v, t)` with one moment tensor per parameter and `t` a
  non-negative `int` (not `bool`) counting updates taken. `t` starts at 0 and is incremented
  before bias correction.
- `lr` is finite and `>= 0`. `beta1`, `beta2` are in `[0, 1)`. `eps` is finite and `> 0`.
  `weight_decay` is finite and `>= 0`.
- Per tensor, with the value `p` before this update and the raw gradient `g`:
  - shrink: `p_decayed = p * (1 - lr * weight_decay)`
  - `m = beta1 * m + (1 - beta1) * g`, `v = beta2 * v + (1 - beta2) * g * g`
  - `m_hat = m / (1 - beta1**t_new)`, `v_hat = v / (1 - beta2**t_new)`
  - `p_new = p_decayed - lr * m_hat / (sqrt(v_hat) + eps)`
- Returns `(new_params, (new_m, new_v, t_new))`, with the input shapes and dtypes.
- Raises `ValueError` for mismatched lengths or shapes, out-of-range or non-finite scalars, and
  malformed state.
- Must not mutate inputs.

## Examples

```python
params, state = adamw_step(
    [torch.tensor([1.0])], [torch.tensor([0.5])], None, lr=0.1, weight_decay=0.5
)
# shrink: 1.0 * (1 - 0.1 * 0.5) = 0.95; Adam step is about 0.1
# params -> [tensor([0.8500])]
```

A zero gradient only decays:

```python
params, state = adamw_step(
    [torch.tensor([2.0])], [torch.tensor([0.0])], None, lr=0.1, weight_decay=0.5
)
# params -> [tensor([1.9000])]
```

## Constraints

Allowed: Python, `torch` tensor operations, and `math`. Not allowed: `torch.optim`, which
contains the optimizer you are checking against. Write the update yourself.

## Done looks like

The public tests pass for a decayed first step, a zero-gradient step and an invalid decay. Hidden
tests compare your trajectory with `torch.optim.AdamW` over several steps and different decay
strengths. They also check the zero-gradient decay factor, that the learning rate scales the
decay, the zero-decay case, the first counter value, input immutability, scale extremes, invalid
arguments and dtype. A failing test names the mistake, such as decay folded into the gradient or
a decay factor that leaves out the learning rate.

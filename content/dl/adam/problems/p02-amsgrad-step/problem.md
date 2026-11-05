# AMSGrad step

Implement `amsgrad_step(params, grads, state, *, lr, beta1=0.9, beta2=0.999, eps=1e-8)`. It is an
Adam update with one change: the denominator uses the running maximum of the second moment, not
the current value. A run of small gradients then cannot shrink the denominator after a large
gradient has passed. This matches `torch.optim.Adam(..., amsgrad=True)`.

## Contract

- `params`, `grads`: lists of floating-point tensors of equal length. `grads[i]` has the same
  shape and dtype as `params[i]`.
- `state`: `None`, or the tuple `(m, v, v_max, t)`. `m`, `v`, `v_max` are lists with one tensor per
  parameter, each the shape of its parameter. `t` is a non-negative `int` (not `bool`).
- `lr` is finite and `>= 0`. `beta1`, `beta2` are in `[0, 1)`. `eps` is finite and `> 0`.
- With `t_new = t + 1`, per tensor, using gradient `g`:
  - `m = beta1 * m + (1 - beta1) * g`
  - `v = beta2 * v + (1 - beta2) * g * g`
  - `v_max = max(v_max, v)`, elementwise, where `v` is the **uncorrected** second moment above.
    `v_max` starts at zero and never decreases. Do not bias-correct `v` before taking the maximum;
    only `v_max` is bias-corrected, as in `torch.optim.Adam(amsgrad=True)`.
  - `m_hat = m / (1 - beta1**t_new)`, `v_max_hat = v_max / (1 - beta2**t_new)`
  - `p_new = p - lr * m_hat / (sqrt(v_max_hat) + eps)`
- Returns `(new_params, (new_m, new_v, new_v_max, t_new))`.
- Raises `ValueError` for mismatched lengths or shapes, out-of-range or non-finite scalars, and
  malformed state.
- Must not mutate inputs.

## Examples

```python
params, state = amsgrad_step([torch.tensor([1.0])], [torch.tensor([2.0])], None, lr=0.1)
# first update: v_max = v, so AMSGrad equals Adam -> params [tensor([0.9000])]
```

With `beta2 = 0.5`, a large gradient followed by a small one:

```python
params, state = amsgrad_step([torch.tensor([1.0])], [torch.tensor([2.0])], None, lr=0.1, beta2=0.5)
params, state = amsgrad_step(params, [torch.tensor([0.1])], state, lr=0.1, beta2=0.5)
# params -> [tensor([0.8388])]   (plain Adam with the same inputs gives a larger step)
```

## Constraints

Allowed: Python, `torch` tensor operations, and `math`. Not allowed: `torch.optim`, which
contains the optimizer you are checking against. Write the update yourself.

## Done looks like

The public tests pass for the first update, the fast-decay second update and an invalid beta.
Hidden tests compare your trajectory with `torch.optim.Adam(amsgrad=True)` on seeded random
tensors, including a sequence where gradients shrink after large ones. They also check input
immutability, malformed state, zero gradients, scale extremes and dtype. A failing test names the
mistake, such as a denominator that forgets the maximum.

# Step decay learning rate

Implement `step_decay_lr(step, base_lr, step_size, gamma)`: the learning rate after
`step` scheduler updates when the rate is multiplied by `gamma` every `step_size`
updates. It must match `torch.optim.lr_scheduler.StepLR` over any run.

## Contract

- `step`: `int`, `step >= 0`. The number of scheduler updates already taken. `bool` is
  not an `int` here.
- `base_lr`: `float`, finite, `base_lr > 0`. The rate before any drop.
- `step_size`: `int`, `step_size >= 1`. Updates between drops.
- `gamma`: `float`, finite, `0 < gamma <= 1`. The multiplicative factor at each drop.
- Returns: a Python `float`, `base_lr * gamma ** (step // step_size)`.
  The rate changes exactly when `step` reaches a multiple of `step_size`.
- Raises: `ValueError` when `step` is negative or not an `int`, when `step_size < 1` or is
  not an `int`, when `base_lr` is not finite and positive, or when `gamma` is not finite
  and within `(0, 1]`.
- Must not mutate inputs. Runs in constant time, however large `step` is.

## Examples

```python
>>> step_decay_lr(0, 0.1, 2, 0.5)
0.1
>>> step_decay_lr(1, 0.1, 2, 0.5)           # still in the first block
0.1
>>> step_decay_lr(2, 0.1, 2, 0.5)           # the first drop happens at step 2
0.05
>>> step_decay_lr(5, 0.1, 2, 0.5)           # floor(5 / 2) = 2 drops
0.025
```

## Constraints

- Allowed: Python arithmetic and the standard library.
- Not allowed: `torch.optim`, including `torch.optim.lr_scheduler`, and any scheduler
  from another library. Compute the closed form.

## Done looks like

The public examples pass. Hidden tests compare with `StepLR` over full runs on seeded
settings, check the boundary on both sides of each drop, a constant schedule when
`gamma` is one, a very large step count evaluated directly (a step-by-step loop does not
finish within the time limit, so it fails), the return type, and the invalid-argument families.

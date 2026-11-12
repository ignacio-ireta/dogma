# Linear warmup then cosine decay

Implement `warmup_cosine_lr(step, base_lr, warmup_steps, total_steps, min_lr)`: the
learning rate after `step` updates when the rate rises linearly for `warmup_steps`
updates and then follows a half-cosine down to `min_lr` at `total_steps`. This is the
common large-model schedule. Build it from the closed form, not from a scheduler class.

## Contract

- `step`: `int`, `step >= 0`. Updates already taken. `bool` is not an `int` here.
- `base_lr`: `float`, finite, `base_lr > 0`. The peak rate, reached at `warmup_steps`.
- `warmup_steps`: `int`, `warmup_steps >= 0`. Length of the linear ramp. `0` means no ramp.
- `total_steps`: `int`, `total_steps > warmup_steps`. The step at which the rate reaches
  `min_lr`.
- `min_lr`: `float`, finite, `0 <= min_lr <= base_lr`. The floor.
- Returns: a Python `float`.
  - `step < warmup_steps`: `base_lr * step / warmup_steps`. Step 0 returns `0.0`.
  - `warmup_steps <= step < total_steps`: with `p = (step - warmup_steps) / (total_steps - warmup_steps)`,
    `min_lr + (base_lr - min_lr) * (1 + cos(pi * p)) / 2`.
    At `step == warmup_steps` this is exactly `base_lr`, so there is no dip and no second peak.
  - `step >= total_steps`: `min_lr`.
- Raises: `ValueError` when `step` is negative or not an `int`; when `warmup_steps` is
  negative or not an `int`; when `total_steps` is not an `int` or is not greater than
  `warmup_steps`; when `base_lr` is not finite and positive; or when `min_lr` is not finite
  and within `[0, base_lr]`.
- Must not mutate inputs.

## Examples

```python
>>> warmup_cosine_lr(0, 1.0, 4, 12, 0.1)     # ramp starts at zero
0.0
>>> warmup_cosine_lr(2, 1.0, 4, 12, 0.1)     # halfway up the ramp
0.5
>>> warmup_cosine_lr(4, 1.0, 4, 12, 0.1)     # the boundary: exactly base_lr
1.0
>>> warmup_cosine_lr(8, 1.0, 4, 12, 0.1)     # halfway down the cosine
0.55
>>> warmup_cosine_lr(12, 1.0, 4, 12, 0.1)    # total_steps: the floor
0.1
```

## Constraints

- Allowed: Python `math` and the standard library.
- Not allowed: `torch.optim`, including `torch.optim.lr_scheduler`, and any scheduler from
  another library. Write the piecewise formula yourself.

## Done looks like

The public examples pass. Hidden tests compare a full run with a `LambdaLR` oracle
driven by an optimizer, on seeded settings. They also check the zero start, the exact
boundary at `warmup_steps`, the floor at and after `total_steps`, the no-warmup case,
the one-step warmup, monotonic shape on each side of the peak, a constant schedule when
`min_lr == base_lr`, extreme magnitudes, the return type, and the invalid-argument families.

Some hidden tests take their inputs from the real datasets `breast_cancer`, `diabetes`, `digits`,
`iris` and `wine` (see `docs/DATASETS.md`).

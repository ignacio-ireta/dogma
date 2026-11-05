# Cosine annealing learning rate

Implement `cosine_annealing_lr(step, base_lr, t_max, eta_min=0.0)`, the learning rate
after `step` scheduler updates under cosine annealing with a floor. The result must
equal what `torch.optim.lr_scheduler.CosineAnnealingLR` produces over a full run, and
you must write the closed form yourself.

## Contract

- `step`: `int`, `step >= 0`. The number of scheduler updates already taken. `bool` is
  not an `int` here.
- `base_lr`: `float`, finite, `base_lr > 0`. The rate at step 0.
- `t_max`: `int`, `t_max >= 1`. The number of updates from `base_lr` to `eta_min`.
- `eta_min`: `float`, finite, `0 <= eta_min <= base_lr`. The floor. Default `0.0`.
- Returns: a Python `float`.
  - For `0 <= step <= t_max`:
    `eta_min + (base_lr - eta_min) * (1 + cos(pi * step / t_max)) / 2`.
  - For `step > t_max`: `eta_min`. The schedule holds its floor. Unlike
    `CosineAnnealingLR`, which keeps oscillating past `T_max`, this contract does not.
- Raises: `ValueError` when `step` is negative or not an `int`, when `t_max < 1` or is
  not an `int`, when `base_lr` is not finite and positive, or when `eta_min` is not
  finite and within `[0, base_lr]`.
- Must not mutate inputs (there are none to mutate; do not keep module-level state).

## Examples

```python
>>> cosine_annealing_lr(0, 0.1, 4)
0.1
>>> cosine_annealing_lr(2, 0.1, 4)          # half way: cos(pi/2) = 0
0.05
>>> cosine_annealing_lr(4, 0.1, 4, eta_min=0.01)
0.01
```

```python
>>> cosine_annealing_lr(1, 0.1, 4)          # (1 + cos(pi/4)) / 2 * 0.1
0.08535533905932738
```

## Constraints

- Allowed: Python `math` and the standard library. `torch` is available in this runtime,
  but only for tensor arithmetic you do not need here.
- Not allowed: `torch.optim`, including `torch.optim.lr_scheduler`, and any
  scheduler or optimizer from another library. Write the formula.

## Done looks like

The public examples pass. Hidden tests compare a full run, step by step, with the
`CosineAnnealingLR` oracle on seeded random settings. They also check the floor
after `t_max`, the single-step schedule, a constant schedule when `eta_min == base_lr`,
symmetry around the midpoint, monotonic decrease, extreme magnitudes of the base rate,
return type, and the invalid-argument families.

# Inverse-square-root schedule

Implement `inverse_sqrt_lr(step, d_model, warmup_steps)`: the learning rate of the schedule
in Vaswani et al. (2017), eq. 3, for the update numbered `step`. The rate rises linearly for
`warmup_steps` updates and then decays with the inverse square root of the step number.

## Contract

- `step`: `int`, `step >= 1`. Updates are numbered from 1, as in the paper. Step 0 has no
  value because the decay term `step ** -0.5` is undefined there. `bool` is not an `int`.
- `d_model`: `int`, `d_model >= 1`. The model width used for scaling.
- `warmup_steps`: `int`, `warmup_steps >= 1`. The step of the peak.
- Returns: a Python `float`,
  `d_model ** -0.5 * min(step ** -0.5, step * warmup_steps ** -1.5)`.
  The two terms are equal at `step == warmup_steps`, which is the peak.
- Raises: `ValueError` when `step < 1` or is not an `int`; when `d_model < 1` or is not an
  `int`; when `warmup_steps < 1` or is not an `int`.
- Must not mutate inputs.

## Examples

```python
>>> inverse_sqrt_lr(1, 4, 4)     # ramp: step * warmup**-1.5 is the smaller term
0.0625
>>> inverse_sqrt_lr(4, 4, 4)     # the peak: both terms equal 0.5
0.25
>>> inverse_sqrt_lr(16, 4, 4)    # decay: step**-0.5 is the smaller term
0.125
```

## Constraints

- Allowed: Python `math` and the standard library.
- Not allowed: `torch.optim`, including `torch.optim.lr_scheduler`, and any scheduler from
  another library. Write the formula.

## Done looks like

The public examples pass. Hidden tests compare a full run with a `LambdaLR` oracle that
uses the scheduler's own counter, on seeded settings. They also check the peak location
and value, the rise and fall on each side of the peak, the decay for very large steps,
the scaling in `d_model`, the one-step warmup, extreme magnitudes, the return type, and the
invalid-argument families, including step 0.

Some hidden tests take their inputs from the real datasets `breast_cancer`, `diabetes`, `digits`,
`iris` and `wine` (see `docs/DATASETS.md`).

# Dynamic loss scaler update rule

Implement a class `DynamicLossScaler`. It holds the loss scale used in fp16 mixed-precision
training and decides, after every backward pass, whether the optimizer step should run and how
the scale should change. The class is pure Python. It does not touch any tensor: the caller
computes `found_inf` (whether any gradient is infinite or NaN) and passes it in.

Why the rule exists: fp16's smallest positive value is about $6\times10^{-8}$, and gradients below about $3\times10^{-8}$ round to zero, so the
loss is multiplied by a scale $S$ before the backward pass and the gradients are divided by $S$
afterwards. A scale that is too large makes gradients overflow to infinity, so an overflowing step
is skipped and the scale is halved. A scale that has been safe for a while is doubled, so the
scale tracks the largest value the gradients can take.

## Contract

- `DynamicLossScaler(init_scale=2.0**16, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000)`
  - `init_scale`: finite `float` greater than 0.
  - `growth_factor`: finite `float` greater than 1.
  - `backoff_factor`: `float` strictly between 0 and 1.
  - `growth_interval`: `int` at least 1.
  - Raises `ValueError` when an argument is out of range or is a `bool`.
- Attributes, always current:
  - `scale`: `float`, the current loss scale. It starts at `init_scale`.
  - `good_steps`: `int`, the number of consecutive steps without overflow since the last change of
    `scale`. It starts at 0.
- `update(found_inf: bool) -> bool`: applies one step's outcome and returns whether the optimizer
  step should be applied.
  - If `found_inf` is `True`: the step is skipped (return `False`), `scale` is multiplied by
    `backoff_factor`, and `good_steps` is reset to 0.
  - If `found_inf` is `False`: the step is applied (return `True`) and `good_steps` increases by 1.
    When `good_steps` reaches `growth_interval`, `scale` is multiplied by `growth_factor` and
    `good_steps` is reset to 0. The count is never kept past a growth, and it restarts from 0 after
    an overflow.
- `scale` is a Python `float`, and the class sets no floor. Each overflow multiplies it by `backoff_factor`, so from the default start of `2**16`, with backoff 0.5, a run of more than 1,090 consecutive overflows underflows to `0.0`.
- Must not mutate its arguments.

## Examples

```python
>>> s = DynamicLossScaler(init_scale=1024.0, growth_interval=3)
>>> s.update(found_inf=False), s.scale, s.good_steps     # first good step
(True, 1024.0, 1)
>>> s.update(found_inf=False); s.update(found_inf=False)  # third good step: the scale doubles
True
>>> s.scale, s.good_steps
(2048.0, 0)
>>> s.update(found_inf=True), s.scale, s.good_steps      # overflow: skip the step, halve the scale
(False, 1024.0, 0)
```

```python
>>> DynamicLossScaler(growth_factor=1.0)
ValueError: growth_factor must be greater than 1
```

## Constraints

- Allowed: the standard library (for example `math`). The class needs no tensor library.
- Not allowed: `torch.amp` and `torch.cuda.amp`, including `GradScaler`. The tests use
  `torch.amp.GradScaler` on CPU as the reference, so reading it would be the answer.

## Done looks like

All public tests pass. Then Submit runs the hidden tests. They cover: agreement with
`torch.amp.GradScaler` on CPU for seeded random overflow sequences (the scale after every step and
the step decision); the growth counter (growth only after a full interval of good steps, and reset by
an overflow); the backoff and skip; custom growth and backoff factors; the default interval of 2000;
a run of 300 overflows that keeps the scale positive; a property check that the scale equals
the initial scale times the product of the factors applied; the `ValueError` rules, including `bool`
arguments and out-of-range factors; and the types of `scale` and `good_steps`.

The hidden tests draw their bulk inputs from the real dataset `online_retail` (see
`docs/DATASETS.md`).

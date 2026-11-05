## Hint 1 — concept
The scaler is a small state machine with two numbers: the scale and a count of good steps since the
last change. Each step either overflows (skip, back off, reset the count) or does not (count it, and grow
when the count reaches the interval). Nothing else changes.

## Hint 2 — formula / approach
On overflow: `scale = scale * backoff_factor`, `good_steps = 0`, return False. On a good step:
`good_steps = good_steps + 1`. If that equals `growth_interval`, then `scale = scale * growth_factor` and
`good_steps = 0`. Return True. Check the constructor arguments with `math.isfinite` and explicit bounds,
and reject `bool` before you check ranges, because `bool` is a subclass of `int`.

## Hint 3 — code shape
Put the four arguments in attributes in `__init__`, after validation, with `scale` stored as a float and
`good_steps` starting at 0. In `update`, use an `if found_inf:` branch that returns early, and then the
good-step branch. Keep the growth check inside the good-step branch, after the counter is incremented.

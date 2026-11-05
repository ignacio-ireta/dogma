## Hint 1 — concept

The schedule rises during warmup and then falls. Both phases come from one expression with a
minimum, so there is no branch on the step. The peak is where the two phases meet.

## Hint 2 — formula / approach

With `k` the update number from 1, `d` the model width and `W` the warmup length, the rate is
`d ** -0.5 * min(k ** -0.5, k * W ** -1.5)`. Compare the two terms: the ramp term is smaller
before `W`, and the decay term is smaller after `W`. Remember that the first update is `k = 1`.

## Hint 3 — code shape

Validate that `step`, `d_model` and `warmup_steps` are ints in their ranges, rejecting bools. Compute
the two terms as separate variables, take their minimum, multiply by `d_model ** -0.5`, and return a
float. Check the peak by hand: at `k = W` both terms should be equal.

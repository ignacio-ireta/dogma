## Hint 1 — concept

Separate the sign regions before thinking about vectorization. The derivative is a local multiplier, not a gradient of a loss.

## Hint 2 — formula / approach

Use a positive mask and a negative mask; reserve a separate value for the zero derivative.

## Hint 3 — code shape

Convert once, validate, then construct outputs with masks or np.where. Avoid writing through the input view.

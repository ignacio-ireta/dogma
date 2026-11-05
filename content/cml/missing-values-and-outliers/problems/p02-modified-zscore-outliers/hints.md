## Hint 1 — concept

The center and the spread are both medians, so a single extreme value cannot move them. The score measures
how far a value sits from the center, in units of the typical deviation from the center.

## Hint 2 — formula / approach

Compute `median = median(x)`, then `mad = median(|x - median|)`. If `mad` is zero, return the values that differ
from the median. Otherwise compute `0.6745 * (x - median) / mad` and compare its absolute value with the threshold,
strictly.

## Hint 3 — code shape

Validate the input and the threshold first. Then compute the two medians with `np.median`, branch on the zero MAD,
and return a boolean NumPy array built from one comparison.

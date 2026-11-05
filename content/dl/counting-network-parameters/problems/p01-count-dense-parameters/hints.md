## Hint 1 — concept

Count scalar slots in one weight matrix and one bias vector before adding layers.

## Hint 2 — formula / approach

For adjacent widths a and b, weights contribute a*b; bias contributes b only if enabled.

## Hint 3 — code shape

Validate the width sequence first. Iterate over neighboring pairs and sum the Python integers.

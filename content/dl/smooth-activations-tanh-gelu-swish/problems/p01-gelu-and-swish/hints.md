## Hint 1 — concept

Write the three scalar definitions first; then preserve the shape while applying them.

## Hint 2 — formula / approach

GELU uses the normal CDF. A stable sigmoid can use exp(-abs(x)) with different expressions on each side of zero.

## Hint 3 — code shape

Flatten only while mapping math.erf, reshape back, and return an explicit three-item tuple.

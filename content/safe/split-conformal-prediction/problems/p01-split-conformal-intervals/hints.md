## Hint 1 — concept

Each calibration point gives one score, its absolute residual. The threshold is one of those scores, picked by a rank that depends on how many points you have and on the miscoverage level alpha. Anything above the threshold falls outside.

## Hint 2 — formula / approach

The rank is k = ceil((n + 1)(1 - alpha)), where the n + 1 counts the test point. The threshold is the k-th smallest score. If k > n, no finite threshold exists and both bounds are infinite. Read alpha as the decimal it was written as (repr gives that text) and compute the rank with exact rational arithmetic, because a float product can round an exact integer up.

## Hint 3 — code shape

Validate alpha and both calibration arrays first, raising ValueError. Compute the absolute residuals and sort them. Compute k with exact rational arithmetic on the decimal value of alpha, then take the ceiling. Set the threshold to infinity when k exceeds n, and otherwise to the entry at index k - 1. Return the test predictions minus and plus the threshold.

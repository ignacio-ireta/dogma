## Hint 1 — concept

Acceptance is a biased coin for the drafted token. Rejection does not return the draft token. It samples the target mass the draft did not cover. Every random number comes from `rng.random()`, in a fixed order.

## Hint 2 — formula / approach

Accept when `u1 < min(1, p[x] / q[x])`. Otherwise take the residual `max(0, p - q)`, divide by its sum, and draw one more uniform `u2`. Return the first index where the cumulative residual exceeds `u2`. Validate the arguments before any arithmetic.

## Hint 3 — code shape

Validate `p`, `q` and `draft_token`, raising `ValueError` on any failure. Compute the threshold, draw `u1` unconditionally, and return early on acceptance. On rejection, build the residual, normalize it, form its cumulative sum with the last entry set to exactly 1, draw `u2`, and locate it with a search over the cumulative array. Put the zero-residual fallback in its own branch.

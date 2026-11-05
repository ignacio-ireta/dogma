## Hint 1 — concept

The cumulative sums cut `[0, 1)` into stretches, one per category, and category `i` owns the
stretch of length `p_i`. A uniform draw lands in exactly one stretch, so finding the category is a
search over the cumulative sums, not a loop over the probabilities.

## Hint 2 — formula / approach

Let `c = cumsum(probs)`. The category for `u` is the number of entries of `c` that are at most `u`,
which is what `np.searchsorted(c, u, side="right")` returns. Rounding can leave `c[-1]` slightly
below 1, so a draw at the very top must still map to a valid category.

## Hint 3 — code shape

Validate first: `probs` must be a non-empty 1-D array of finite, non-negative values whose sum is
within `1e-9` of 1, and `n` must be non-negative. Then draw `u = rng.random(n)` once, search the
cumulative sums with `side="right"`, and clip the result to the last category.

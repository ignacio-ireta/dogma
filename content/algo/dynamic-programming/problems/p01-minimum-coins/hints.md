## Hint 1 — concept

Ask what the answer for a total depends on. If you know the fewest coins for every smaller total,
can you say the fewest coins for this total? Work out which smaller total the last coin leaves.

## Hint 2 — formula / approach

Let `best[v]` be the fewest coins that make the value `v`, with `best[0] = 0`. For `v >= 1`,
`best[v]` is one more than the smallest `best[v - c]` over the coins `c` that are at most `v`.
Fill the table for `v = 1, 2, ..., amount` in that order, so each read is already final. Mark an
unreachable value with a number larger than any real answer, such as `amount + 1`.

## Hint 3 — code shape

Make the table with `best = [0] + [unreachable] * amount`. Then two loops: the outer loop over
`v` from 1 to `amount`, and the inner loop over the coins. Inside, update `best[v]` when
`best[v - coin] + 1` is smaller. Return `best[amount]`, mapped to `-1` when it still holds the
unreachable value.

## Hint 1 — concept

Evaluate a forecaster the way it will be used: fit on the past, score on the future, then move the
boundary forward. Each split is one boundary. A gap is rows you deliberately throw away between the two
sides, so that information arriving late cannot leak across the boundary.

## Hint 2 — formula / approach

For each split, the cutoff is `c = initial_train + j * step`. Training is `[0, c)` when expanding, or
`[c - initial_train, c)` when rolling. The test window is `[c + gap, c + gap + horizon)`. Keep the splits
for which the test window ends at or before `n`.

## Hint 3 — code shape

Validate the parameters and the length first. Then start `cut` at `initial_train` and loop while
`cut + gap + horizon <= n`. Inside the loop, build the training range with `np.arange` (the start is 0 or
`cut - initial_train`), build the test range with `np.arange(cut + gap, cut + gap + horizon)`, append the
pair, and add `step` to `cut`.

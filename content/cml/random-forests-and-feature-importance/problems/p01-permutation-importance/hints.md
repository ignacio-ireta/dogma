## Hint 1 — concept

Permutation importance asks what happens to the score when the model can no longer read one column. Shuffling the column keeps its values but breaks their link to the target, so the drop from the unshuffled baseline measures how much the model relied on it. The protocol fixes the random draws, so every number is reproducible: one generator, used for every feature and repeat in a fixed order. Measure the spread over repeats, but use the population formula for it.

## Hint 2 — formula / approach

Baseline: `metric(y, predict(X))`. For feature j and repeat r, draw a fresh permutation of the row indices, build a copy of X whose column j is taken from the permuted rows, and record `baseline − metric(y, predict(X_perm))`. The mean drop of feature j is the mean over its repeats. The standard deviation is the population one, dividing by the number of repeats, which means numpy's default `ddof=0`.

## Hint 3 — code shape

Validate X, y, n_repeats and seed first. Create `rng = np.random.default_rng(seed)` once, before any loop. Compute the baseline once. Allocate a `(d, n_repeats)` array of drops and fill it with nested loops, with the feature loop outside and the repeat loop inside. In each step, copy X, assign `X_arr[perm, j]` to column j of the copy, and score it, checking that predict returned one value per row. Finish with the row-wise mean and standard deviation of the drop array.

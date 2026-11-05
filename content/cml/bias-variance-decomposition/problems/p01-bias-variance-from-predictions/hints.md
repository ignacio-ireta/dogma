## Hint 1 — concept

Each model's error at a test point splits into two parts: the average model's distance from the truth (bias), and
how much the models disagree with each other (variance). Label noise adds a constant on top, because it is
independent of anything the model does.

## Hint 2 — formula / approach

With `B` models and `m` points: `pbar = mean over models of predictions`. Then `bias2_j = (pbar_j - f_j)**2`,
`var_j` is the mean over models of `(predictions - pbar)**2` (divisor `B`), and the expected error at each point is
`bias2_j + var_j + noise_var`. Average each quantity over the `m` points.

## Hint 3 — code shape

Validate shapes and values first. The conceptual reductions are model means and `ddof=0` variances
along axis 0, followed by averages over test points. For very large common offsets, work with deviations
from one prediction before averaging, and rescale deviations before squaring. Restore scale only after
the normalized calculation, then return Python floats.

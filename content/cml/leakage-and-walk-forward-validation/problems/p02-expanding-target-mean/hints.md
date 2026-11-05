## Hint 1 — concept

A target encoding is a feature, so it may only use what was known when the row arrived. Walk the rows in
time order and keep a running mean and count for each category. Read the encoding before incorporating
the current target, so the current row cannot encode itself.

## Hint 2 — formula / approach

For row `i` with category `c`, the encoding is the previous category mean if the count is positive,
and NaN otherwise. Only after writing that entry, incorporate `target[i]` into the mean and increase
the count. A weighted mean update avoids overflowing a raw sum of finite targets.

## Hint 3 — code shape

Use two dictionaries keyed by category, plus a float64 output array filled with NaN. Loop over
`enumerate(categories)`. Read the category state, write the output when its count is positive, then
update both dictionaries. Use arithmetic that preserves a finite mean at large target magnitudes.

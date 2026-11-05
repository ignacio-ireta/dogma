## Hint 1 — concept

Two decisions are made for each column: how to fill the gaps, and whether the gaps are worth recording.
The fill is estimated from observed values only. The indicator records a fact about the row, not about the
value, so it is kept even when the fill is good.

## Hint 2 — formula / approach

For each named column, find the missing mask first, on the input frame. If the mask has any `True`,
compute the median (or mean) of the observed values, convert the column to float64, and fill. Then append
one `int8` indicator per such column, after the original columns, in the order of `numeric`.

## Hint 3 — code shape

Validate the strategy and the column names. Build the masks in a dictionary. Copy the frame. For each
column with a gap, assign the filled column to the copy. Then loop over the names again and append each
indicator with `astype(np.int8)` on the mask. Return the copy.

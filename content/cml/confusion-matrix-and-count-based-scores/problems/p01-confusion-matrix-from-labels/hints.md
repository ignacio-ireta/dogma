## Hint 1 — concept

The confusion matrix is a table of tallies. Each sample is one (true, predicted) pair, and the answer is how many samples land in each cell. Decide first which label goes on which axis, and which labels get a row at all.

## Hint 2 — formula / approach

Build the label order first: the sorted union of both arrays, or the caller's `labels`. Map each label to its position with a dictionary. For each pair (t, p), add one to the cell at row `index[t]` and column `index[p]`. A label outside the order has no position, so skip that pair. Check the input lengths before you start counting.

## Hint 3 — code shape

Compute `order`, then `index = {label: i for i, label in enumerate(order)}`. Create an integer array of shape `(k, k)`. Loop over `zip(y_true, y_pred)`, keep only pairs where both labels are in `index`, and add one to the matching cell. Raise `ValueError` before the loop for mismatched lengths, empty input or duplicate labels.

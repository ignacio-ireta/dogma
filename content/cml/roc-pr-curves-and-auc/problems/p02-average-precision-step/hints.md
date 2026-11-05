## Hint 1 — concept

Average precision is the area under the precision-recall curve, summed in steps. Each new threshold adds the recall it gains, and that gain is weighted by the precision reached at the threshold. Lowering the threshold through a tie adds a whole group of samples at once, so ties have to be handled as one step.

## Hint 2 — formula / approach

Sort the samples from the highest score to the lowest, and take cumulative sums of the labels. Evaluate only at the end of each run of equal scores. At each evaluated threshold, precision is TP divided by the number flagged, and recall is TP divided by the total number of positives. AP is the sum over thresholds of the recall increment times the precision, with the recall before the first threshold taken as 0.

## Hint 3 — code shape

Validate first, including the case with no positives. Sort with `np.argsort(-values, kind="stable")`, take `np.cumsum` of the sorted labels, and keep the positions where the next sorted score differs. Build `recall_before` by prepending 0 to the recall array, then return `float(np.sum((recall - recall_before) * precision))`.

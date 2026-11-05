## Hint 1 — concept

AUC is a probability over pairs: pick one positive and one negative at random, and ask how often the positive scores higher. Ranking the scores gives the same count without comparing every pair. Only the order of the scores matters, so any increasing transform leaves the answer unchanged.

## Hint 2 — formula / approach

Sort the scores with a stable argsort. Give each sample its 1-based rank. Samples with equal scores occupy a run of consecutive ranks, and they must all receive the average of that run. Then sum the ranks of the positives and subtract the smallest possible sum, n_pos (n_pos + 1) / 2. Divide by n_pos times n_neg.

## Hint 3 — code shape

Validate first, including NaN, labels other than 0 and 1, and the single-class case. Sort, then mark where the sorted values change, and use `np.cumsum` on those marks to label each run. Give each run its midrank, which is (first rank + last rank) / 2. Write the midranks back to the original positions with `ranks[order] = ...`, then compute the formula from the positives' ranks.

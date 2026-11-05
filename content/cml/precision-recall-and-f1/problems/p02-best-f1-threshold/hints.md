## Hint 1 — concept

Lowering the threshold flags more samples: some of them are positive, which raises recall, and some are negative, which hurts precision. Sweeping the threshold over the scores in order, and keeping running counts, finds the best F1 without retesting each threshold from scratch.

## Hint 2 — formula / approach

Sort the samples from the highest score to the lowest. The samples flagged at any threshold form a prefix of that order, so cumulative sums of the labels give TP for every prefix. A tie must be flagged together, so only the last position of each run of equal scores is a candidate. Then F1 = 2 TP / (2 TP + FP + FN) at each candidate. Take the candidate with the largest F1, and among equal values the one that comes first in the descending order.

## Hint 3 — code shape

Validate the input first, including NaN and infinite scores and labels other than 0 and 1. Build `order = np.argsort(-values, kind="stable")` and the sorted labels. Use `np.cumsum` on the sorted labels for TP, and `np.arange(1, n + 1)` for the number flagged. Keep only the candidates where the next sorted score differs. Compute F1 as one division and return the first index of its maximum as a pair of Python floats.

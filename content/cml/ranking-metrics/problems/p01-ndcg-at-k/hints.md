## Hint 1 — concept

NDCG compares the ranking you produced with the best ranking you could have produced from the same
items. The gain is the relevance, and the discount shrinks with depth. The hard part is ties: equal
scores do not define an order, so the value must not depend on the order of the input items.

## Hint 2 — formula / approach

Sort by score, descending. Group the equal scores into runs. A run that occupies positions a..b
contributes its mean relevance times the sum of the discounts 1 / log2(r + 1) for r in a..b, counting
only r within k. Prefix sums of the discounts make each run cost O(1). IDCG is the same sum over the
relevance sorted descending. If IDCG is 0, return 0.0.

## Hint 3 — code shape

Validate first, including NaN, negative relevance and `k`. Then `np.argsort(-scores, kind="stable")`,
a boolean mark where the sorted scores change, `np.cumsum` on the marks to label each run, and
`np.bincount` with weights for the sum of relevance per run. Keep a prefix array of the discounts.
Combine the pieces into DCG, and compute IDCG from `np.sort(relevance)[::-1]`.

## Hint 1 — concept

Retrieval evaluation asks three questions of one ranked list: how many of the top k are relevant
(precision and recall), how early the first relevant item appears (reciprocal rank), and whether the
relevant items sit high throughout the list (average precision). Each one has its own denominator, and
the denominators are where the mistakes hide.

## Hint 2 — formula / approach

Mark each ranked position as a hit or a miss. Precision at k divides the hits in the top k by k. Recall
divides them by the number of distinct relevant ids. Reciprocal rank uses the first hit in the whole
list. For average precision, walk the top k, and at each hit add the running precision (hits so far over
the position). Divide the sum by min(k, number of relevant ids).

## Hint 3 — code shape

Copy the ranked list and turn the relevant ids into a set first, so duplicates collapse. Check the
duplicate ranked ids, the empty relevant set and `k` before any counting. Build `hits` with a
comprehension. Count the top-k hits with `sum(hits[:k])`. Use `next(...)` over `enumerate(hits)` for the
first hit. Finish with one loop over `enumerate(hits[:k], start=1)`. Return a dict with the four keys.

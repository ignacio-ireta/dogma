## Hint 1 — concept

The median is the middle of the sorted values, so it is decided by the two values next to the middle.
Keep the smaller half of the data in one structure and the larger half in another. The top of each is then
the value next to the middle, and a median needs only those two tops.

## Hint 2 — formula / approach

Use a max-heap for the lower half (store negatives in `heapq`) and a min-heap for the upper half. Insert each
value into the correct half, then move one element across if the sizes differ by more than one. If the two
sizes are equal, the median is the mean of the two tops. Otherwise it is the top of the larger heap, which
is the lower one.

## Hint 3 — code shape

Keep two lists, `lower` and `upper`, and a `medians` list. Per value: push to `lower` as `-value` unless it
is greater than `-lower[0]`, in which case push to `upper`. Rebalance with `heappop` and `heappush` on
the other list. Append the mean of the two tops when the sizes are equal, and `-lower[0]` otherwise. Compute the mean without overflowing a same-sign sum: for ordered tops `a <= b` of the same sign, use `a + (b - a) / 2`; with opposite signs, `(a + b) / 2` is safe.
Return `medians`.

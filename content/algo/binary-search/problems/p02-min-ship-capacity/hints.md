## Hint 1 — concept

The smallest capacity that works is a boundary: capacities below it fail and capacities at or above
it succeed. A larger capacity never needs more days, so you can halve the range of capacities, the
same way you halve a range of positions.

## Hint 2 — formula / approach

The answer is at least the heaviest parcel, because one truck must carry it. It is at most the total
weight, because one truck can carry everything. For a candidate capacity, a greedy pass counts the
days: fill the current day with parcels until the next one would exceed the capacity, then start a
new day. If the count is at most `days`, the candidate works and the answer is that candidate or
smaller. Otherwise the answer is larger.

## Hint 3 — code shape

```python
lo, hi = ..., ...  # the heaviest parcel, and the total weight
while lo < hi:
    mid = ...  # a candidate capacity inside [lo, hi)
    if ...:  # the greedy pass for mid needs at most days
        hi = ...
    else:
        lo = ...
return lo
```

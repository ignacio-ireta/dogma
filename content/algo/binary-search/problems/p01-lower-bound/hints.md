## Hint 1 — concept

The answer is the boundary between the elements smaller than the target and the elements at or
above it. Because the list is sorted, every element on one side of the boundary is smaller than
every element on the other. You can therefore halve the range where the boundary can be, at each
step.

## Hint 2 — formula / approach

Keep a half-open interval `[lo, hi)` that contains the answer. Start with `lo = 0` and
`hi = len(sorted_values)`. Look at the middle index `mid`. If `sorted_values[mid] < target`, the
answer is after `mid`, so `lo = mid + 1`. Otherwise `mid` might be the answer, so `hi = mid`. When
`lo == hi`, the interval is empty and `lo` is the answer.

## Hint 3 — code shape

```python
lo, hi = 0, len(sorted_values)
while lo < hi:
    mid = ...  # the middle index of [lo, hi)
    if ...:  # sorted_values[mid] is smaller than target
        lo = ...
    else:
        hi = ...
return lo
```

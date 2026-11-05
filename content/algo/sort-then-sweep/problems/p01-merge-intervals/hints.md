## Hint 1 — concept

Two intervals can only merge if they are next to each other in some order. What order puts every candidate
pair next to each other, so that one pass over the list can see all merges?

## Hint 2 — formula / approach

Sort the intervals by start. Walk through them and keep the last merged interval. A new interval merges into it
when its start is less than or equal to the last merged end, because touching at a shared point counts. When it
merges, the new end is the larger of the two ends, since an interval can lie inside another. Otherwise start a new
merged interval. Validate every interval before you sort.

## Hint 3 — code shape

```python
for start, end in intervals:
    if start > end:
        raise ValueError(...)
ordered = sorted(...)  # a new list, sorted by start
merged = []
for start, end in ordered:
    if merged and start <= ...:  # shares a point with the last merged interval
        merged[-1] = (..., max(..., end))
    else:
        merged.append((start, end))
return merged
```

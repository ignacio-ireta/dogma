## Hint 1 — concept

A hash map (a Python `dict`) finds a key in O(1) on average, so one pass over the tokens can count
all of them. After counting, you only have to rank the distinct tokens. Counting and ranking are
two separate steps.

## Hint 2 — formula / approach

For each token, add one to its entry, starting from 0 when the token is new. Then sort the
`(token, count)` pairs by count descending and token ascending. Return the first `k` pairs, or an
empty list when `k <= 0`. Python compares tuples element by element, so the key can encode both
rules at once.

## Hint 3 — code shape

```python
counts: dict[str, int] = {}
for token in tokens:
    counts[token] = ...  # your increment, starting from 0 for a new key
ranked = sorted(counts.items(), key=...)  # your key: count descending, then token ascending
return ...  # the first k pairs, or [] when k <= 0
```

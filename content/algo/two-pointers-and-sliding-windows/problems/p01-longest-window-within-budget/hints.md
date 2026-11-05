## Hint 1 — concept

Think of a window that slides along the list. Its right edge moves forward once per step. Its left edge
moves forward only when the window is too expensive. Both edges only move forward, so the work is O(n).
Everything depends on the costs being non-negative: that is what makes "too expensive" stay too expensive
as the window grows.

## Hint 2 — formula / approach

Keep `window_sum` for `costs[left..right]`. For each `right`, add `costs[right]`. While the sum is above
`budget`, subtract `costs[left]` and advance `left`. The window `costs[left..right]` is then the longest
valid window that ends at `right`. Record it only when it is strictly longer than the best so far, so ties
keep the smallest start.

## Hint 3 — code shape

```python
best_length, best_start = 0, 0
left = 0
window_sum = 0
for right, cost in enumerate(costs):
    window_sum += ...  # grow on the right
    while window_sum > budget:  # shrink on the left until valid
        window_sum -= ...
        left += 1
    length = right - left + 1
    if length > best_length:  # strictly greater keeps the smallest start
        ...
return best_length, best_start
```

## Hint 1 — concept

The window moves right, and the answer is the largest value inside it. Ask which values can never be the
maximum again: a value that is smaller than a newer value, or a value that has left the window. What must you
remember about the others?

## Hint 2 — formula / approach

Keep a deque of indices. Its values never increase from front to back, so the front is the answer. For each new
index: drop the front index if it has left the window; drop the back indices whose values are not larger than the
new value; push the new index. Once the first window is complete, each step appends the front value. Each index is
pushed once and popped once, so the total work is O(n).

## Hint 3 — code shape

```python
window: deque[int] = deque()
result = []
for i, value in enumerate(values):
    if window and ...:  # the front index has left the window
        window.popleft()
    while window and ...:  # back values that the new value makes useless
        window.pop()
    window.append(i)
    if i >= k - 1:
        result.append(values[window[0]])
return result
```

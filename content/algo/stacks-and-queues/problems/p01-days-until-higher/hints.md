## Hint 1 — concept

Each reading waits until a strictly larger reading arrives. Only the readings still waiting need to be
remembered. When a new reading arrives, which waiting readings does it resolve, and where in the list do those
waiting readings sit?

## Hint 2 — formula / approach

Keep the indices of the waiting readings on a stack, last in first out. For each new reading, pop every waiting
index whose value is strictly smaller than the new value. For each popped index, the answer is the new index
minus that index. Then push the new index. A reading that is still on the stack at the end has no answer, and its
entry stays 0. Each index is pushed once and popped once, so the total work is O(n).

## Hint 3 — code shape

```python
result = [0] * len(values)
waiting: list[int] = []
for index, value in enumerate(values):
    while waiting and ...:  # pop each waiting index that this value resolves
        earlier = waiting.pop()
        result[earlier] = ...
    waiting.append(index)
return result
```

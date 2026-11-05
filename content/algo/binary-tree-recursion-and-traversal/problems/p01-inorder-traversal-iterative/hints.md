## Hint 1 — concept

In-order means the left subtree first, then the node, then the right subtree. A recursive walk does
this with the call stack. Ask what the call stack remembers at each level, and what you would have
to remember yourself if the calls were not allowed.

## Hint 2 — formula / approach

Keep a list of nodes that you have started but not finished. Walking down the left side pushes
each node, and the last one pushed is the leftmost. When you cannot go further left, the top of the
list is a node whose left subtree is done: emit it, then move to its right child and repeat. The
walk ends when the list is empty and the current pointer is `None`.

## Hint 3 — code shape

Use a `while` loop with two inner steps: an inner `while` that pushes along the left spine, and a
pop that emits and moves right. Keep one variable for the current node, and check both the stack
and the current node in the loop condition:

```python
result = []
stack = []
node = root
while ...:
    while ...:
        ...
    node = ...
    result.append(...)
    node = ...
return result
```

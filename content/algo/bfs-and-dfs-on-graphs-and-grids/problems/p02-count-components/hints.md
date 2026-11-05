## Hint 1 — concept

A component is a region that one search can fill completely. If you scan every cell in order and start a search at each
open cell you have not yet covered, each search is exactly one component. What must the search remember, so that it never
starts twice in the same island?

## Hint 2 — formula / approach

Keep a `seen` grid of booleans. Scan the grid row by row. When you meet an open cell that is not seen, add one to the
count, mark the cell seen, and flood its component: pop a cell, look at its up, down, left and right neighbours, and
mark and push each open unseen neighbour that is inside the grid. Flood with an explicit stack, not recursion, because
an island can be tens of thousands of cells long.

## Hint 3 — code shape

The outer scan and the inner flood are two loops. The skeleton is:

```python
components = 0
for r in range(rows):
    for c in range(cols):
        if grid[r][c] == "#" or seen[r][c]:
            continue
        components += 1
        seen[r][c] = True
        stack = [(r, c)]
        while stack:
            row, col = stack.pop()
            # for each of the four neighbours: check bounds, wall and seen, then mark and push
return components
```

Check the first example by hand before you run the tests: the count is 2.

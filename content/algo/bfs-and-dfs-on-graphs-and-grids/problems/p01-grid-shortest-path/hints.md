## Hint 1 — concept

Every move costs 1, so the cells reached first are the closest ones. Think about which container gives you
cells in the order of their distance from the start, and what happens if you forget to remember where you have been.

## Hint 2 — formula / approach

Keep a queue of cells and a record of distances, with -1 for "not reached yet". Start with the start cell at distance 0.
Pop a cell, then for each of its four neighbours that is inside the grid, open, and not yet recorded, set its distance to
one more and enqueue it. Record the distance when you enqueue, not when you pop. If the goal is recorded, its distance is
the answer. If the queue empties first, the answer is -1.

## Hint 3 — code shape

Check the bounds before you read the grid, so a neighbour at index -1 is never used. The loop has this shape:

```python
distance[start_row][start_col] = 0
queue = deque([(start_row, start_col)])
while queue:
    row, col = queue.popleft()
    for dr, dc in MOVES:
        nr, nc = row + dr, col + dc
        # bounds check, wall check, already-recorded check, then record and enqueue
    ...
return -1
```

Check the worked example by hand first: on the grid of three rows with a wall pair in the middle row, the goal is four
moves away.

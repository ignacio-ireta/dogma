# Shortest path in a grid with walls

Implement `shortest_path_length(grid, start, goal)`. The grid is a list of strings. `.` is an open cell and `#` is a
wall. A move goes to one of the four neighbouring cells (up, down, left, right) if that cell is open and inside the
grid. Each move costs 1. Return the fewest moves from `start` to `goal`, or `-1` if no route exists.

## Contract

- `grid`: a non-empty `list` of `str`. Every row has the same length, which is at least 1. Each character is `.` or
  `#`. A grid that is not a list raises `TypeError`, and so does a row that is not a `str`. An empty grid, rows of unequal length or any other character raise `ValueError`.
- `start` and `goal`: `tuple` values `(row, column)` of Python `int` values (not `bool`). Row 0 is the top row.
- Both cells must lie inside the grid, and both must be open. Otherwise raise `ValueError`.
- Returns: a Python `int`. `0` when `start == goal`. `-1` when `goal` cannot be reached.
- The grid is never changed, since it is a list of strings.

## Examples

```python
>>> shortest_path_length(["...", "##.", "..."], (0, 0), (2, 2))
4        # right, right, down, down: (0,0) (0,1) (0,2) (1,2) (2,2)
>>> shortest_path_length([".#.", "##.", "..."], (0, 0), (2, 2))
-1       # the start cell is boxed in by walls
>>> shortest_path_length(["..", ".."], (1, 1), (1, 1))
0
>>> shortest_path_length(["..#", "#.."], (0, 0), (0, 2))
ValueError  # goal is a wall
```

## Constraints

- Allowed: built-in types and the standard library.
- Not allowed: `heapq`. A priority queue is not needed, because every move costs 1.
- The search must not revisit a cell along a second route. The hidden tests include a 700 by 700 open grid,
  which a search without a visited record cannot finish within the time limit.

## Done looks like

The public tests check the examples, a start-equals-goal case and a wall-goal error. **Submit** adds hidden tests
that compare the length with an independent frontier-by-level search on seeded random grids, and they cover: open
and fully walled grids; a goal sealed off by walls; a route that a depth-first search would find long before the
shortest one; a single row and a single column; the start on the goal; boundaries of the grid; the type rules; the
value rules; and the large open grid, which must finish within the time limit.

The hidden tests draw their bulk inputs from the real dataset `digits` (see `docs/DATASETS.md`).

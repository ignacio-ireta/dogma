# Count connected components in a grid

Implement `count_components(grid)`. The grid is a list of strings, where `.` is an open cell and `#` is a wall. Two open
cells belong to the same component when a path of up, down, left and right moves joins them through open cells. Return the
number of components. Each island of open cells counts once, no matter how many cells it has.

## Contract

- `grid`: a non-empty `list` of `str`. Every row has the same length, at least 1. Each character is `.` or `#`.
  A grid that is not a list raises `TypeError`, and so does a row that is not a `str`. An empty grid, rows of unequal length or any other character raise `ValueError`.
- Returns: a Python `int`, at least 0. A grid with no open cells returns `0`.
- The grid is never changed.

## Examples

```python
>>> count_components(["..#", "..#", "##."])
2        # one 2x2 island at the top left, and the single cell at the bottom right
>>> count_components([".#.", "#.#", ".#."])
5        # diagonal neighbours do not connect: each of the five open cells is its own component
>>> count_components(["###", "###"])
0
>>> count_components(["."])
1
```

## Constraints

- Allowed: built-in types and the standard library.
- Not allowed: the `sys` module. Do not change the interpreter's recursion limit.
- A recursive depth-first search is not allowed for this problem. The hidden tests include a single open region of
  about 45,000 cells in a snake shape, which needs far deeper recursion than Python allows by default.

## Done looks like

The public tests check the examples. **Submit** adds hidden tests that compare your count with a union-find oracle on
seeded random grids, and they cover: all open; all walls; diagonal neighbours that must stay apart; single row and column
grids; a snake-shaped island deeper than the default recursion limit; islands touching the grid border; the return type;
the validation rules; and a large grid of many small islands, which must finish within the time limit.

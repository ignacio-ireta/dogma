# Largest value in each window

Implement `sliding_window_max(values, k)`. A window of width `k` slides across `values` one position at a time,
starting at the left end. Report the largest value inside each window, from the first window to the last.

## Contract

- `values`: a `list` of `float`, possibly empty. Every entry is a finite number.
- `k`: an `int`. It must satisfy `1 <= k <= len(values)`.
- Returns: a `list` of `float` with `len(values) - k + 1` entries. Entry `i` is the largest value among
  `values[i]` through `values[i + k - 1]`. Each entry is one of the input values, unchanged.
- Raises: `ValueError` when `k < 1` or `k > len(values)`. An empty list is covered: every `k >= 1` is larger
  than 0.
- Must not mutate `values`.
- Complexity: **O(n) time** for `n = len(values)`, using a double-ended queue (`collections.deque`) or another linear-time method. The hidden
  tests include a list of 200,000 values with `k = 50,000`. Taking the maximum of every window from scratch
  does about `n * k` comparisons there, and it will not finish in time.

## Examples

```python
>>> sliding_window_max([4.0, 2.0, 12.0, 11.0, -5.0], 2)
[4.0, 12.0, 12.0, 11.0]
>>> sliding_window_max([7.0, 7.0, 1.0, 0.5], 2)
[7.0, 7.0, 1.0]
>>> sliding_window_max([3.0, -2.0, 8.0], 3)
[8.0]
```

```python
>>> sliding_window_max([1.0, 2.0], 0)
ValueError: k must be between 1 and len(values)
>>> sliding_window_max([], 1)
ValueError: k must be between 1 and len(values)
```

## Constraints

- Allowed: built-in types (`list`, `float`, `int`), `len`, `range`, `enumerate`, `max` and `min` on small
  slices, `collections.deque`, and the standard library.
- Not allowed: `heapq`. A heap gives O(n log n), and the point of this problem is the deque. Not allowed: a
  scan over every window that takes the maximum of each one, which is O(n * k).

## Done looks like

All public tests pass, then **Submit** runs the hidden tests. They compare your answer against a brute-force
maximum on small and medium random lists, with ties and with several window widths. The edge families are:
a window of width 1, a window as wide as the whole list, a maximum that must leave the window and be replaced
by a smaller value, a maximum that stays inside for exactly `k` positions, equal maxima across a slide, the
invalid widths (zero, negative, larger than the list, and an empty list) that must raise `ValueError`, an input
that must not change, and a list of 200,000 values that must finish within the time limit.

The hidden tests draw their bulk inputs from the real datasets `elnino` and `sunspots` (see
`docs/DATASETS.md`).

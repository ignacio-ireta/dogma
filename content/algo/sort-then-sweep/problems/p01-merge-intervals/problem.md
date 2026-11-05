# Merge overlapping intervals

Implement `merge_intervals(intervals)`. Each interval is a closed range `(start, end)` of whole numbers, such as
the time span of a job. Combine every group of intervals that share at least one point into one interval, and
return the combined intervals in order.

## Contract

- `intervals`: a `list` of `tuple[int, int]`, possibly empty. Each tuple is `(start, end)` and represents the
  closed range `[start, end]`. The input may be in any order and may contain duplicates and nested ranges.
- Two intervals **merge** when they share at least one point. They overlap, or one ends exactly where the other
  starts (touching: `(1, 3)` and `(3, 5)` merge into `(1, 5)`). Intervals that do not share a point stay apart:
  `(1, 3)` and `(4, 5)` remain two intervals, even though no whole number lies strictly between 3 and 4.
- Returns: a `list` of `tuple[int, int]`, sorted by start. No two returned intervals share a point.
- Raises: `ValueError` when any interval has `start > end`. The check applies to every interval, wherever it
  appears in the list.
- Must not mutate `intervals`, either the list or its tuples.
- Complexity: **O(n log n) time** for `n = len(intervals)`: sort once, then one pass. The hidden tests include a
  list of 200,000 intervals. Comparing every pair of intervals, and repeating until nothing changes, does about
  `n**2` comparisons there, and it will not finish in time.

## Examples

```python
>>> merge_intervals([(5, 7), (1, 2), (2, 4), (9, 9)])
[(1, 4), (5, 7), (9, 9)]
# (1, 2) and (2, 4) touch at 2, so they merge. (5, 7) starts after 4, so it stays apart.
>>> merge_intervals([(3, 5), (1, 3)])
[(1, 5)]
>>> merge_intervals([(1, 3), (4, 5)])
[(1, 3), (4, 5)]
```

```python
>>> merge_intervals([(0, 10), (2, 3), (4, 4)])
[(0, 10)]
>>> merge_intervals([])
[]
>>> merge_intervals([(2, 1)])
ValueError: an interval must have start <= end
```

## Constraints

- Allowed: built-in types (`list`, `tuple`, `int`), `len`, `sorted`, `max` and `min`, and the standard library.
- Not allowed: a comparison of every pair of intervals. Not allowed: changing `intervals` in place.

## Done looks like

All public tests pass, then **Submit** runs the hidden tests. They compare your answer against a brute-force
coverage check on small random lists, where intervals touch, nest and repeat. The edge families are: an empty
list, a single interval, duplicate intervals, single-point intervals, nested intervals, a chain of touching
intervals, intervals one whole number apart (which must stay apart), unsorted input, negative numbers, an invalid
interval anywhere in the list, an input that must not change, output tuples of plain `int`, and a list of 200,000
intervals that must finish within the time limit.

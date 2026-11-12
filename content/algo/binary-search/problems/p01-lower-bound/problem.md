# Lower bound in a sorted list

Implement `lower_bound(sorted_values, target)`. Return the first index `i` at which
`sorted_values[i] >= target`. If no element is at least `target`, return `len(sorted_values)`. The
list can be long, so the search must take a logarithmic number of steps, not a scan.

## Contract

- `sorted_values`: a `list` (or any sequence) of `int` or `float` in non-decreasing order. It may be
  empty, may contain duplicates, and may contain negative numbers. Tests never pass an unsorted list.
- `target`: an `int` or a `float`, compared numerically with the elements.
- Returns: an `int` `i` with `0 <= i <= len(sorted_values)`. Every element before index `i` is
  smaller than `target`. When `i < len(sorted_values)`, `sorted_values[i]` is at least `target`.
- Raises: nothing in this problem.
- Reads `sorted_values` with `len()` and indexing only. Do not copy, sort or iterate over it.
- Must not mutate `sorted_values`.

## Examples

```python
>>> lower_bound([1, 2, 2, 2, 5], 2)    # the run of 2s starts at index 1
1
>>> lower_bound([1, 3, 5], 4)          # 4 would sit between 3 and 5
2
```

```python
>>> lower_bound([1, 3, 5], 0)          # smaller than every element
0
>>> lower_bound([1, 3, 5], 9)          # larger than every element: len, not len - 1
3
>>> lower_bound([], 7)                 # empty list
0
```

## Constraints

- Allowed: built-in types, `typing`, `collections.abc`, and the standard library otherwise.
- Not allowed: `bisect` (as an import or as attribute access such as `bisect.bisect_left`) and
  `numpy` (as an import or as attribute access such as `np.searchsorted`). Write the search yourself.

## Done looks like

All public tests pass, then **Submit** runs the hidden tests. They compare your answer with a
reference on random sorted lists that contain duplicates, integers and floats. The edge families are:
the first index of a run of equal values, a target below every element, a target above every element,
the empty list, negative and float values (including a float target that must not be truncated), an
integer result within bounds, inputs that must not change, and a large list where a scan reads far
too many elements.

The hidden tests draw their bulk inputs from the real datasets `digits` and `iris` (see
`docs/DATASETS.md`).

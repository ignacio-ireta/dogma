# Running median with two heaps

Implement `running_median(values)`. It returns the median of every prefix of `values`: element `i` of the
result is the median of `values[0..i]`, after the i-th value has been added. Keep two heaps, so that each
insertion costs O(log n), and the whole input costs O(n log n).

## Contract

- `values`: a sequence of finite real numbers (ints or floats), possibly empty.
- Result: a list of `len(values)` Python floats. Element `i` is the median of `values[: i + 1]`: the middle
  value if the count is odd, and the mean of the two middle values if the count is even.
- Empty `values` gives `[]`.
- Values may be as large as `1e308`. Compute the mean of the two middle values without overflow: for
  `1e308` and `1e308` it is `1e308`.
- Raises `ValueError` if any value is NaN or infinite.
- Must run in O(n log n) time. Must not mutate its input.

## Examples

```python
>>> running_median([2, 1, 3, 4])
[2.0, 1.5, 2.0, 2.5]
# prefixes: [2] -> 2, [2, 1] -> 1.5, [2, 1, 3] -> 2, [2, 1, 3, 4] -> (2 + 3) / 2 = 2.5
```

```python
>>> running_median([7, 7, 7])
[7.0, 7.0, 7.0]
```

```python
>>> running_median([])
[]
```

## Constraints

- Allowed: the Python standard library (`heapq`, `math`) and the list type.
- Not allowed: `statistics` and `bisect` (both listed in `banned_imports`). Do not sort each prefix, and do
  not keep one sorted list and insert into it. Use two heaps.

## Done looks like

All public tests pass, then **Submit** runs the hidden tests. They compare each prefix median with
`statistics.median` on small random inputs of ints and of floats. They check the even-count rule with an
odd sum, which must give a half, and the odd-count rule. They check ascending, descending and alternating
inputs, duplicates, negative numbers and large magnitudes with small differences, a single value, and an
empty input. They check that NaN and infinite values raise `ValueError`, that the result is a list of
floats, and that the input is unchanged. A 60,000-value input must finish within a time limit, which a
sort of each prefix does not meet.

The hidden tests draw their bulk inputs from the real datasets `elnino` and `sunspots` (see
`docs/DATASETS.md`).

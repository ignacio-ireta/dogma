# Days until a higher reading

Implement `days_until_higher(values)`. `values` is a list of sensor readings in time order. For each position,
count how many steps forward you must move until you reach a reading that is strictly larger than the one at
that position. If no later reading is larger, the answer for that position is `0`.

## Contract

- `values`: a `list` of `float`, possibly empty. Every entry is a finite number. Negative values are allowed.
- Returns: a `list` of `int` with the same length as `values`. For position `i`, the entry is `j - i`, where `j`
  is the smallest index greater than `i` with `values[j] > values[i]`. It is `0` when no such `j` exists.
- A later reading that is equal to `values[i]` is not larger, so it does not end the wait.
- Raises: nothing for a list of finite numbers.
- Must not mutate `values`.
- Complexity: **O(n) time** and O(n) extra space, for `n = len(values)`. The hidden tests include a list of
  200,000 readings. Checking every later position for every index does about `n**2 / 2` comparisons there, and it
  will not finish in time.

## Examples

```python
>>> days_until_higher([5.0, 3.0, 8.0, 8.0, 2.0, 9.0])
[2, 1, 3, 2, 1, 0]
# 5.0 waits 2 steps for 8.0; 8.0 at index 2 is not beaten by the equal 8.0 at index 3,
# so it waits 3 steps for 9.0.
```

```python
>>> days_until_higher([1.0, 1.0, 1.0])
[0, 0, 0]
>>> days_until_higher([-2.5, -1.0, -3.0])
[1, 0, 0]
>>> days_until_higher([])
[]
```

## Constraints

- Allowed: built-in types (`list`, `int`, `float`), `len`, `range`, `enumerate`, `min`, `max`, and the standard
  library.
- Not allowed: a scan over every later position for each index. It can be correct on small inputs and still
  fails the time limit on large ones.

## Done looks like

All public tests pass, then **Submit** runs the hidden tests. They compare your answer against a brute-force
scan on small and medium random lists, which contain ties and negative readings. The edge families are: empty
input, a single reading, runs of equal readings, a run that falls with no higher reading after it, a long run
that a single higher reading resolves at once, negative readings compared by value, an input that must not
change, entries that must be plain `int`, and a list of 200,000 readings that must finish within the time limit.

The hidden tests draw their bulk inputs from the real dataset `elnino` (see `docs/DATASETS.md`).

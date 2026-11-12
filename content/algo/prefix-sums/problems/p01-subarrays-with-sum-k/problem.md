# Count subarrays with sum k

Implement `count_subarrays_with_sum(nums, k)`. Count the contiguous, non-empty subarrays of `nums` whose
entries sum to exactly `k`. The entries may be negative, so a sliding window does not apply: a window can
get smaller and then larger again. Use prefix sums instead. Keep a count of the earlier prefix sums, and
in one pass look up how many of them equal the current prefix sum minus `k`.

## Contract

- `nums`: a `list` of Python `int`, possibly empty. Negative, zero and positive values are all allowed.
  Values are exact: nothing is rounded.
- `k`: a Python `int`, possibly negative or zero.
- Returns: a Python `int`, not a float and not a NumPy integer. It is the number of index pairs
  `(i, j)` with `0 <= i < j <= len(nums)` such that `sum(nums[i:j]) == k`. Two subarrays are different
  when they cover different positions, even if their values match.
- Raises `TypeError` when `k` or any entry of `nums` is not a Python `int`. A `bool`, a `float`, a string,
  `None` and a NumPy integer are all rejected.
- Complexity: O(n) time and O(n) extra space, for `n = len(nums)`. The hidden tests include a list of
  200,000 entries. Checking every pair of positions does about 2 x 10^10 steps there, and it will not finish
  in time.
- Must not mutate `nums`.

## Examples

```python
>>> count_subarrays_with_sum([1, 1, 1], 2)
2          # [1, 1] at positions 0-1, and [1, 1] at positions 1-2
>>> count_subarrays_with_sum([1, -1, 1, -1], 0)
4          # [1, -1] at 0-1, [-1, 1] at 1-2, [1, -1] at 2-3, and the whole list
>>> count_subarrays_with_sum([-2, 3, -1], -1)
1          # only [-1] at position 2; [-2, 3, -1] sums to 0 and is not counted
>>> count_subarrays_with_sum([], 5)
0
>>> count_subarrays_with_sum([1, 2], 1.0)
TypeError  # k must be a Python int
```

## Constraints

- Allowed: built-in types and the standard library (a `dict` is the intended tool).
- Not allowed: `numpy` and `scipy`, and any nested loop over every start and end position. A quadratic
  scan can be correct on small inputs and still miss the time limit on large ones.

## Done looks like

All public tests pass, then **Submit** runs the hidden tests. They compare your count with a brute-force
count on small lists that mix negative and zero values, and with an independent position-based count on
larger lists. The edge families they cover: an empty list; a target of zero with many zero entries; a
target that no subarray reaches; a subarray that starts at the first position; negative targets and
all-negative lists; very large values, where the count must stay exact; the return type; rejected types;
inputs that must not change; and a 200,000-entry list that must finish within the time limit.

The hidden tests draw their bulk inputs from the real datasets `breast_cancer` and `elnino` (see
`docs/DATASETS.md`).

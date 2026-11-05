# Longest window within a budget

Implement `longest_window_within_budget(costs, budget)`. `costs` is a sequence of non-negative
integers, such as the cost of each step of a route. Find the longest run of **consecutive** entries
whose sum is at most `budget`, and report where it starts.

## Contract

- `costs`: a `list` of `int`, possibly empty. Every entry is `>= 0`.
- `budget`: an `int` that is `>= 0`.
- Returns: a `tuple` `(length, start)` of two Python `int`s.
  - `length`: the number of entries in the longest window `costs[start:start + length]` whose sum is
    `<= budget`.
  - `start`: the index of the first entry of that window. Among windows of the same maximum length,
    return the smallest `start`.
  - If no window is valid, return `(0, 0)`. That happens when `costs` is empty or every entry is
    greater than `budget`.
- Raises: `ValueError` when any entry of `costs` is negative, or when `budget` is negative.
- Must not mutate `costs`.
- Complexity: **O(n) time** and O(1) extra space, for `n = len(costs)`. The hidden tests include an input
  with `n = 200,000`. A solution that checks every start and extends each window to the right does
  about `n**2 / 2` steps there, and it will not finish in time. One left pointer and one right pointer,
  each moving only forward, do about `2n` steps.

## Examples

```python
>>> longest_window_within_budget([1, 2, 3, 4, 5], 9)
(3, 0)     # [1, 2, 3] sums to 6 and [2, 3, 4] sums to 9; both have length 3, so start 0 wins
>>> longest_window_within_budget([4, 1, 1, 1, 9], 3)
(3, 1)     # [1, 1, 1] sums to exactly 3, which is allowed
>>> longest_window_within_budget([0, 0, 7, 0], 0)
(2, 0)     # zero-cost entries cost nothing, so [0, 0] is valid
>>> longest_window_within_budget([5, 6], 4)
(0, 0)     # every single entry exceeds the budget
>>> longest_window_within_budget([1, 2], -1)
ValueError: budget must be non-negative
```

## Constraints

- Allowed: built-in types (`list`, `int`, `tuple`), `range`, `len`, `min` and `max`, and the standard
  library.
- Not allowed: a quadratic scan over all windows. It can be correct on small inputs and still fails
  the time limit on large ones.

## Done looks like

All public tests pass, then **Submit** runs the hidden tests. They compare your answer against a
brute-force scan on small random lists and against an independent prefix-sum computation on larger ones.
The edge families are: empty input, a budget of zero, a window that sums to exactly the budget, ties
between windows of equal length, no valid window, a long run at the end of the list, negative costs and
negative budgets, inputs that must not change, and a list of 200,000 entries that must finish within the
time limit.

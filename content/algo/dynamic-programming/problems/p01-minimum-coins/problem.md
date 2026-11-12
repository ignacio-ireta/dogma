# Minimum coins for change

A transit kiosk gives change from a fixed set of coin denominations, and each coin type can be
used any number of times. The kiosk wants to give the amount with as few coins as possible.
Implement `min_coins(coins, amount)`. It returns the fewest coins whose values sum exactly to
`amount`, or `-1` when no combination of the coins can make it.

Solve this with the dynamic-programming system: a table where `best[v]` is the fewest coins that
make the value `v`. Each entry depends only on smaller values, so one pass in increasing order of
`v` answers every amount up to the one asked for.

## Contract

- `coins`: a `list` of Python `int`, possibly empty. Each coin value must be at least 1. Order is
  arbitrary, and a value may appear more than once. The list must not be changed.
- `amount`: a Python `int` that is at least 0.
- Returns: a Python `int`. It is the minimum number of coins (counting repeats) whose values sum
  to `amount`. It is `0` when `amount` is `0`, even with no coins. It is `-1` when the amount
  cannot be made. Impossible is a normal answer, so it is not an exception.
- Raises `TypeError` when `coins` is not a list, when a coin is not a Python `int` (a `bool`,
  `float` or string is rejected), or when `amount` is not a Python `int`.
- Raises `ValueError` when a coin is less than 1, or when `amount` is negative.
- Complexity: O(amount x number of coins) time. The hidden tests include an amount of 50,000,
  which must finish within the time limit.

## Examples

```python
>>> min_coins([1, 2, 5], 11)
3          # 5 + 5 + 1
>>> min_coins([1, 3, 4], 6)
2          # 3 + 3; the greedy choice 4 + 1 + 1 uses three coins
>>> min_coins([2], 3)
-1         # only even values can be made
>>> min_coins([5, 10], 0)
0
>>> min_coins([1, 2], -1)
ValueError
```

## Constraints

- Allowed: built-in types and the standard library.
- Not allowed: `functools`, including `functools.lru_cache` and `functools.cache`. Build the table
  yourself so that the order of evaluation is visible in your code.
- A recursion that tries every coin at every amount without remembering results takes exponential
  time. A memoised recursion can exceed Python's recursion limit when the amount is large, so the
  hidden tests use an amount of 50,000.

## Done looks like

All public tests pass, then **Submit** runs the hidden tests. They compare your answer with a
breadth-first search over sums for small random inputs, and with an independent coin-by-coin table
for larger ones. Each call has a 5-second time box. A table of amounts finishes in milliseconds,
so a call that hits the box is doing repeated work. The edge families they cover: an empty coin list; an amount of zero; impossible
amounts; a greedy trap; repeated and unsorted coins; large amounts that must not overflow the
recursion limit; the return type; rejected types and values; and inputs that must not change.

The hidden tests draw their bulk inputs from the real dataset `online_retail` (see
`docs/DATASETS.md`).

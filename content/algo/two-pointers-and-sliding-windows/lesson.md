---
title: Two pointers and sliding windows
minutes: 12
min_viable: "Read The system and the Worked example, then solve p01."
---

## The system

A window is a contiguous range `costs[left..right]`. Two pointers mark its ends, and both only move
forward. The template has one shape:

```python
left, state = 0, empty
best = ...
for right, x in enumerate(items):
    state += x                          # grow: include items[right]
    while state violates the rule:      # shrink: drop from the left until valid again
        state -= items[left]
        left += 1
    best = record(best, right - left + 1)   # the window items[left..right] is valid here
```

Four facts about the template:

- **Invariant:** after each step, `items[left..right]` is valid, and it is the longest valid window
  that ends at `right`.
- **Work:** each index enters the window once and leaves it at most once, so the total is O(n).
- **Shrink is a `while`:** one large item can force several removals in one step.
- **Count:** the length is `right - left + 1`, after the shrink.

The same template covers the usual variants:

| Variant | State | Shrink while | Notes |
|---|---|---|---|
| Longest window with sum at most `budget` | running sum | sum > budget | Costs must be non-negative |
| Fixed size `k` | running sum or count | size > k | Remove exactly one item, so `if` is enough |
| At most `k` distinct values | count per value (dict) | distinct > k | Delete a key when its count reaches 0 |
| Longest run with no repeats | count per value (dict) | some count > 1 | Same shape as the row above |

## Worked example

`costs = [1, 2, 3, 4, 5]`, `budget = 9`.

| right | add | removed from left | window | sum | length | best (length, start) |
|---|---|---|---|---|---|---|
| 0 | 1 | none | [1] | 1 | 1 | (1, 0) |
| 1 | 2 | none | [1, 2] | 3 | 2 | (2, 0) |
| 2 | 3 | none | [1, 2, 3] | 6 | 3 | (3, 0) |
| 3 | 4 | 1 | [2, 3, 4] | 9 | 3 | (3, 0), a tie, so the earlier start stays |
| 4 | 5 | 2, 3 | [4, 5] | 9 | 2 | (3, 0) |

At `right = 4` the sum reaches 14, and two removals are needed. This is why the shrink is a `while`.
The answer is `(3, 0)`.

## Why it works

Non-negative costs make two things true. Growing a window never lowers its sum, and shrinking it never
raises its sum. So for a fixed `right`, the valid values of `left` form one block, and the smallest
valid `left` never moves backward as `right` grows. The pointers therefore only move forward, and the
window found at each step is the longest valid one ending there. Every valid window is considered.

The pointers move at most `2n` times in total, so the work is O(n). This is an amortized bound (see
the amortized-analysis chapter of Cormen et al.).

For ties, record a window only when it is **strictly** longer. Right ends are visited in increasing
order, so the first window of each maximum length has the smallest start.

## Signal vs. noise

- **Signal:** the invariant, the `while` in the shrink step, the forward-only pointers, and the
  requirement that the rule be monotone in the window. Those decide whether the template is valid.
- **Signal:** the tie rule. "Strictly longer" versus "at least as long" is a common source of wrong
  start indices.
- **Noise for now:** prefix sums with binary search solve the non-negative sum problem in O(n log n).
  A valid alternative, not the target technique.

## Where it breaks

- **Negative values.** The shrink step relies on monotonicity, and negative costs break it. Take
  `costs = [5, -4]` with `budget = 3`. The window `[5]` is over budget, so the template removes the 5
  and moves `left` past index 0. At `right = 1` it reports length 1 starting at index 1. An exhaustive
  search finds `[5, -4]` with sum 1, length 2, starting at 0. The problem rejects negative costs for
  this reason.
- **Using `if` where `while` is needed.** For this problem the answer still comes out right, because an
  invalid window never measures longer than the best valid window found so far. The window is still
  invalid at those steps, though, so the invariant no longer holds. Write the `while`.
- **Recording before shrinking.** If the length is read before the shrink, an invalid window is counted.
- **Forgetting to remove the cost.** Moving `left` without subtracting `costs[left]` leaves the sum too
  high, so the window never becomes valid.
- **Off-by-one length.** `right - left` is one short of the item count.

## Interview angle

"Longest subarray whose sum is at most `k`, with non-negative values." Answer shape: name the window and
the monotone rule first, then the template, then O(n) with the reason (each index enters and leaves once).
Ask whether values can be negative, because the answer changes: then the window is not monotone, and a
prefix-sum approach is needed. Follow-ups: the fixed-size window (remove one item, no `while`), and "at
most `k` distinct characters", which needs a count dictionary.

## Connections

- **Requires:** `algo/hash-maps-and-sets`. The distinct-count variants keep a count per value in a dict.
- **Unlocks:** binary search over prefix sums, and monotone-queue problems, where the same forward-only
  pointers hold a window maximum.

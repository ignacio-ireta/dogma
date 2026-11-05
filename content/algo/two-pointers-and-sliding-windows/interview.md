# Interview notes: Two pointers and sliding windows

## Q1. Longest subarray whose sum is at most `k`. How do you solve it?

**60-second answer.** I name the window first: a contiguous range with a running sum. Before I code, I ask
whether the values can be negative, because the approach depends on it. With non-negative values, adding an
item never lowers the sum and removing one never raises it, so the smallest valid left end only moves forward.
I grow the window on the right, shrink it on the left with a `while` until the sum is within `k`, and record
the length after each step. Each index enters and leaves at most once, so the time is O(n). Ties are broken by
recording only strictly longer windows, which keeps the smallest start.

**Follow-ups.**
- *What if values can be negative?* The window is no longer monotone, so the shrink can discard a valid window.
  The approach then uses prefix sums with an ordered structure over them, O(n log n), not two pointers.
- *What about a fixed size?* Keep the window at size `k`. Each step adds one item and removes one, so an `if` is enough.

## Q2. Longest substring with at most `k` distinct characters. How?

**60-second answer.** It is the same template with a different state. I keep a count per character in a dictionary,
and the window is valid while the number of keys with a positive count is at most `k`. The shrink step removes the
character at `left`, decrements its count, and deletes the key when the count reaches zero. The count of distinct
keys is the only thing I check, so each step is constant time and the whole pass is O(n).

**Follow-ups.**
- *Why delete the key at zero?* Otherwise `len(counts)` counts characters that are no longer in the window.
- *How would you test it?* Use a string where the best window is at the end, a string with one character,
  and `k = 0`.

## Common mistakes
- Shrinking with `if` when one new item can force several removals.
- Moving `left` without subtracting the removed item, so the sum grows and the window never becomes valid.
- Counting `right - left` instead of `right - left + 1`.
- Using `>=` for the budget check, which excludes windows whose sum equals the budget.
- Recording the window before the shrink, so an invalid window is counted.
- Applying the template to values that can be negative, without checking.

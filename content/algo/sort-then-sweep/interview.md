# Interview notes: Sort, then sweep

## Q1. Given a list of closed time intervals, merge the ones that overlap or touch, and return them in order.

**60-second answer.** I validate that each start is at most its end, then sort by start. I keep the last merged
interval. A new interval merges into it when its start is at most the last end, because an interval that starts
exactly where another ends shares that point. The merged end is the larger of the two ends, so a nested interval does
not shrink the range. Otherwise I start a new group. The sort costs O(n log n), and the pass is O(n). I check empty
input, one interval, duplicates, a long interval that comes first, and a chain of touching intervals.

**Follow-ups.**
- *What if the intervals are half-open?* If merging only overlapping pairs, use `start < last_end`: `[1, 3)` and `[3, 5)` share no point. If the contract asks for a minimal union, adjacent half-open pieces may still merge into `[1, 5)`.
- *Why not compare every pair?* That is O(n squared). After sorting, only the open group can overlap a new interval.

## Q2. Given sessions as half-open spans, what is the largest number active at one moment?

**60-second answer.** I turn each session into a start event with +1 and an end event with -1, and sort the events.
At equal times, the end comes first, because a session that ends at t is not active at t. I keep a running total and
its maximum. This is the minimum number of rooms for meetings, and it is O(n log n) because of the sort. A quadratic
check, which counts the sessions containing each start, is correct but too slow for large inputs.

**Follow-ups.**
- *What if I use two sorted lists, one of starts and one of ends?* A pointer over the ends works, because it counts how
  many sessions have already ended before each start. It gives the same answer in O(n log n).
- *What changes for closed sessions?* Starts come first at a tie, because a session ending at t is still active at t.

## Common mistakes
- Forgetting to sort, so the merge only compares neighbours in input order.
- Sorting by end instead of start, which lets a short interval hide the start of a long one.
- Replacing the merged end instead of taking the maximum, which shrinks a nested interval.
- Treating whole-number neighbours as touching, for example merging `(1, 3)` and `(4, 5)`.
- Applying starts before ends at a tie for half-open sessions, which counts touching sessions as overlapping.

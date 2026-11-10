---
title: "SQL LAG and LEAD"
minutes: 12
min_viable: "Read The system and Where it breaks, then solve p01 using Hint 1 if needed."
---

## The system

`LAG(x, k, default)` returns the value of `x` from the row $k$ positions earlier in the same
partition, in the window's ordering. `LEAD` looks forward. Four things define the result before
you write any SQL:

1. **Partition**: the sequence that the lookback stays inside (`PARTITION BY account_id`).
2. **Order**: the sequence positions (`ORDER BY usage_date`, or time plus an identifier).
3. **Offset** $k$: how many rows back. The default is 1.
4. **Default**: the value when no such row exists. The default is `NULL`.

Each output row keeps its own columns and gains the neighbour's value. Row count does not change.

| Question | Answer in this topic |
|---|---|
| Does `LAG` look across partitions? | No. It stops at each partition's first row |
| What does the first row of a partition get? | The default, `NULL` unless you pass one |
| Does a missing calendar day count? | No. The predecessor is the previous recorded row |

## Worked example

Account 7 has these recorded days: 1 Feb with 30 calls, 2 Feb with 45, 4 Feb with 45, and 5 Feb with
12. Ordered by date, `LAG(calls)` gives NULL, 30, 45, 45. The changes are NULL, $+15$, $0$, $-33$.
The 4 Feb row compares with 2 Feb, two calendar days earlier, because 3 Feb has no row. A `LAG`
of two rows would instead compare with 1 Feb, and a `LAG` with a default of 0 would show 0 in the
first row, where the answer is NULL.

## Why it works

A window function computes its value from the partition after the rows are grouped and ordered, and
it does not collapse rows. `LAG` is a fixed positional lookup inside that sorted sequence. Its
meaning depends entirely on the order: change the order and the "previous" row changes too.

Ties in the ordering column make the order undefined. If two views share a timestamp, the previous
row of the second one may be either view. A unique tiebreaker, such as an identifier, makes the
sequence unique. In a sessionization problem the tiebreaker decides which tied row carries a
"session starts here" flag, even when the session numbers come out the same.

## Signal vs. noise

- **Signal:** write the partition, the order and the default before the expression. Most wrong
  answers come from a missing partition or an order with ties.
- **Signal:** `LAG` is applied after `WHERE`. Filtering rows first changes what the predecessor is.
- **Noise for now:** frame clauses. `LAG` and `LEAD` ignore the frame; they need only the order.
- **Noise for now:** dialect-specific shortcuts for time differences. Compare timestamps with
  subtraction and an interval.

## Where it breaks

- **Missing partition.** Without `PARTITION BY`, the previous row can belong to another account.
- **Filtering before LAG.** `WHERE kind = 'success'` before `LAG` makes the failed events vanish
  from the sequence, so a success can look like the predecessor of another success.
- **Ties.** Two events with equal timestamps have no fixed previous row. Add a unique column to the
  order. The final `ORDER BY` also needs it, or the output order is arbitrary.
- **NULL in the ordering column.** `NULLS LAST` and `NULLS FIRST` are dialect choices. State them.
- **Default of 0.** `COALESCE(LAG(x), 0)` turns "no predecessor" into a real value of 0. Keep NULL.
- **Integer sums.** A `SUM` of integer flags can return a wider type than `BIGINT` in DuckDB. Cast
  it when the contract names a type.

**Milliseconds.** On 1,200 views from 40 visitors, about one gap in seven (14%) sits at 30 minutes give or
take up to a second. A gap measured with `DATE_DIFF('second', ...)` counts whole-second boundaries
crossed, not the exact gap, so it gets 60 views (5%) wrong: the fraction decides the answer. It finds 233 session starts where the
answer has 293. Measure the gap as an exact interval: `viewed_at - previous_at > INTERVAL 30
MINUTE`.

## Interview angle

Start by naming the partition, the order and the tiebreak. Then write one `LAG` for the previous value
and one comparison for the change or the gap. For sessions, the answer has three steps: the gap per
row, a flag where the gap exceeds the threshold, and a running sum of the flags. Mention the strict
comparison at the threshold. Full answers and follow-ups are in `interview.md`.

## Connections

This topic builds on window ranking: the same partition-and-order mechanism, with positions
that refer to neighbours instead of to ranks. The next topics are running totals and frames, and
streaks (gaps and islands), where the running sum of a flag is the main tool. `LEAD` is the mirror
image and is used for "time until the next event".

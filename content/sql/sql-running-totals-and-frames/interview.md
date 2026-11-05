# Interview notes — SQL running totals and frames

## Q1. How would you compute a running balance per account?
**60-second answer.** Partition by account so the total restarts for each account. Order by the
event time and a unique posting identifier, so each row has one position. Write the frame as
ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW, so each row's total is the sum up to its own
position. Cast the sum back to the schema's integer type if the engine widens it.
**Follow-ups.** Why not just ORDER BY the timestamp? Tied rows then share the same default
RANGE total, so one row can show the balance of a later row. What if the ledger is huge? The
engine sorts each partition, so the partition key and the order key drive the cost.

## Q2. How would you compute a seven-day moving average of daily sales, where some days have no row?
**60-second answer.** Name the window first: seven calendar days ending on the current day. Use
a RANGE frame on the date with an interval of six days preceding, partitioned by the store. Then
the window follows the calendar, and missing days do not shift it. Report the number of days
that actually contributed, so a reader can see when the average rests on fewer days.
**Follow-ups.** Why not ROWS BETWEEN 6 PRECEDING? It takes the last seven rows, which can span
more than seven days when a day is missing. Should missing days count as zero? That is a
business decision; the query above does not zero-fill, and it says so.

## Q3. What is the difference between ROWS, RANGE and GROUPS?
**60-second answer.** ROWS counts physical rows in the order. RANGE uses the order value, so an
interval such as seven days is meaningful, and peers are included at CURRENT ROW. GROUPS counts
peer groups. Choose the unit that matches the meaning of the window, then state it.
**Follow-ups.** Which is the default? RANGE, when an ORDER BY is present. It is a problem only when the
order has ties: then tied rows share the whole tie group's total.

## Common mistakes
- Relying on the default frame when the order has ties.
- Using ROWS for a calendar window over data with gaps.
- Leaving a running total without a unique tiebreaker, so the result depends on input order.
- Forgetting that SUM over BIGINT is a wider type in DuckDB.

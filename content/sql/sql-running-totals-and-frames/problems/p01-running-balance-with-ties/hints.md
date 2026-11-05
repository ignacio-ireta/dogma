## Hint 1 — concept

A running total needs an order in which every row has its own position. Ask what happens to two
postings with the same timestamp when the order only looks at the timestamp.

## Hint 2 — formula / approach

Partition by account. Order by the timestamp and then the posting identifier, so that each posting
has a unique position. The balance after a posting is the sum over the positions up to it: a frame from the first row
to the current row. Once the order is unique, the default frame gives the same totals, so the
choice of frame is yours, and an explicit `ROWS` frame is the clearest way to state it.

## Hint 3 — code shape

Compute the position with `ROW_NUMBER` and the running sum with `SUM ... OVER`, using the same
`PARTITION BY` and `ORDER BY` in both. The frame is the last clause in the `OVER` parentheses.
Remember the cast on the sum before you select it.

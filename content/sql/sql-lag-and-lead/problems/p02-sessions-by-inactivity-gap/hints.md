## Hint 1 — concept

Sessions are runs of views separated by long pauses. Mark the views that begin a run, then number
the runs with a running count. Both steps need the same view order, per visitor.

## Hint 2 — formula / approach

Use `LAG(viewed_at)` over a window partitioned by visitor and ordered by time and then by the view
identifier. A view starts a session when there is no previous view or when the gap is greater than
30 minutes. A running `SUM` of those start flags, ordered the same way, gives the session number.

## Hint 3 — code shape

Use three stages: a CTE with the lagged timestamp, a CTE with the boolean start flag, and a final
select with the windowed sum cast to BIGINT. Use the same `ORDER BY viewed_at, view_id` inside
both windows, and end with the stated output order.

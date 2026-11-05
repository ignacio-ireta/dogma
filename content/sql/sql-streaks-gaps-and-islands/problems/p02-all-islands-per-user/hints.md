## Hint 1 — concept

A streak is a maximal run of consecutive days. Each day either continues the previous day's run or
starts a new one. Decide that per day first, then number the runs with a running total.

## Hint 2 — formula / approach

Remove duplicate visits. For each day, look at the previous active day of the same user with `LAG`.
If the difference is exactly one day, flag 0; otherwise (including the first day of a user) flag 1.
A running `SUM` of the flags, ordered by date and restarted per user, gives every day of a run the same
number. Group by user and that number, then take `MIN` and `MAX` of the dates and `COUNT` of the days.

## Hint 3 — code shape

Write four `WITH` steps: distinct user-day pairs, the flagged rows with a `CASE` on
`visit_date - LAG(visit_date) OVER (...)`, the running sum with `ROWS BETWEEN UNBOUNDED PRECEDING AND
CURRENT ROW`, and a final `GROUP BY user_id, streak_no`. Cast the count to `BIGINT` and order by
`user_id, island_start`.

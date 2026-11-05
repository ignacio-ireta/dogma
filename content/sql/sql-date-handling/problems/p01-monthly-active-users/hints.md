## Hint 1 — concept

Decide the grain before writing any join. The answer has one row per month of the range, not one row per
event. When a month has no events, which side of the join must keep its row?

## Hint 2 — formula / approach

Build the calendar first: one row per month, generated from a series of dates, and cast to DATE. Then
`LEFT JOIN` the events onto it with a half-open test: the event is at or after the month's first day, and
before the first day of the next month. A timestamp compared with `BETWEEN` on the last day can include
the next month's first instant. Count distinct users, not rows.

## Hint 3 — code shape

```sql
WITH months AS (
    SELECT ...                      -- one row per month of the range, typed as DATE
)
SELECT m.month, COUNT(...) AS active_users
FROM months AS m
LEFT JOIN events AS e ON ...        -- half-open month test on event_ts
GROUP BY ...                        -- one row per month
ORDER BY ...;
```

## Hint 1 — concept

The answer has one row per channel, so `GROUP BY channel` gives the structure. The open question is
how to count and sum only some of the rows inside each group. Put the condition inside the
aggregate, so one pass over the table produces every column.

## Hint 2 — formula / approach

A count of matching rows: count only the matches, with a `CASE` that gives 1 or NULL, or with
`FILTER (WHERE ...)`. A revenue by device: sum a `CASE` that returns the revenue for that device.
Three things to check: `COUNT(column)` counts FALSE values too, because FALSE is not NULL; a `SUM`
over no matching non-NULL value is NULL, not 0; and `//` truncates while `/` between two integers
returns a fraction. Divide as DOUBLE, and guard the zero denominator with `NULLIF`.

## Hint 3 — code shape

```sql
SELECT channel,
       COUNT(...)            AS sessions,
       ...                   AS converted_sessions,
       ROUND(... / NULLIF(...), 4) AS conversion_rate,
       COALESCE(SUM(... ), 0.0)    AS mobile_revenue,
       COALESCE(SUM(... ), 0.0)    AS desktop_revenue
FROM sessions
GROUP BY ...
ORDER BY ...;
```

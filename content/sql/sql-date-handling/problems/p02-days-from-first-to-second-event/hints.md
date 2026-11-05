## Hint 1 — concept

Each user's events form a list in time order. The question asks about only its first two entries, and about
calendar days, not elapsed time. Which two events does the question name, and what does "one day apart" mean
for two timestamps?

## Hint 2 — formula / approach

Number each user's events in time order, so position 1 is the first event and position 2 is the second. Keep
the users who have both positions, and take the difference of their dates. A difference of two TIMESTAMPs is
an interval, and its day field counts whole elapsed days, so 23:59 to 00:01 the next day reads as 0.

## Hint 3 — code shape

```sql
WITH ranked AS (
    SELECT user_id, event_ts, ROW_NUMBER() OVER (...) AS position
    FROM events
)
SELECT ...                              -- the first event joined to the second event of the same user
FROM ... INNER JOIN ... ON ...          -- an inner join keeps only users with two events
ORDER BY ...;
```

## Hint 1 — concept

A streak is a run of consecutive calendar days. Inside a run, every day is one day after the previous
one, so something that moves with the dates can be constant inside each run. Before you look for that
thing, remove duplicate visits: two visits on one day are one active day.

## Hint 2 — formula / approach

For each user, number the distinct days with `ROW_NUMBER()` ordered by the date. Subtract that number
from the date. DuckDB rejects `DATE` minus `BIGINT`, so cast the row number to `INTEGER` first. Within one run the date rises by one each step and the row number also rises by one, so
the difference stays the same. A gap makes the difference change. Group by the user and that difference
to get one row per run, with `MIN` and `MAX` for the ends and `COUNT` for the length. Then keep the row
with the largest length per user, breaking ties on the earliest start.

## Hint 3 — code shape

Use a `WITH` chain of four steps: distinct user-day pairs, the numbered pairs with the difference as a
key, the grouped runs, and the ranked runs. Rank the runs with `ROW_NUMBER() OVER (PARTITION BY user_id
ORDER BY streak_days DESC, streak_start ASC)`. Filter to rank 1, cast the length to `BIGINT`, and order
the output by `user_id`.

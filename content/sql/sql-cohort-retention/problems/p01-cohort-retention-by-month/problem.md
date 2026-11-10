# Monthly cohort retention table

A language-practice app logs one row for every activity a learner records. A learner can record
several activities in the same month. The product team wants a monthly retention table: for each
cohort month (the month in which a learner first recorded activity), what share of that cohort was
active in each later month, including the cohort's own month.

Write one read-only DuckDB query that returns that table.

## Contract

Table `activity` has columns:

| Column | DuckDB type | Rules |
|---|---|---|
| user_id | BIGINT | Non-null |
| event_at | TIMESTAMP | Non-null; many rows may share a user and a month |

The log may hold exact duplicate rows, for example a write that the client retried. A duplicate
row adds nothing: every count below is of distinct users.

Definitions, for every user `u`:

- **Cohort month** `c_u`: the first day of the calendar month of `u`'s earliest `event_at`, as a `DATE`.
- **Cohort size** `N_c`: the number of users whose cohort month is `c`.
- **Period** `p`: the whole number of calendar months from the cohort month to an activity month.
  The cohort's own month is period 0. Measure it in calendar months, not days.
- **Active users** `A(c, p)`: the number of **distinct** users in cohort `c` with at least one event in
  the calendar month `c + p`. A user with three events in one month counts once.
- **Retention rate** `R(c, p) = A(c, p) / N_c`, rounded to 4 decimal places.

Rows of the result:

- One row for every cohort month that has at least one user, and for every period from 0 up to the
  number of months between that cohort month and the latest activity month in the whole table, inclusive.
  A period with no active user is still a row, with `active_users = 0` and `retention_rate = 0.0`.
- A cohort month with no new users has no rows.

Output columns, in this order, with these types:

| Column | DuckDB type |
|---|---|
| cohort_month | DATE |
| period | BIGINT |
| active_users | BIGINT |
| cohort_size | BIGINT |
| retention_rate | DOUBLE |

- `retention_rate` is `ROUND(active_users / cohort_size, 4)`. Ties at the fifth decimal may round
  either way; the grader accepts both.
- Final output order is `cohort_month ASC, period ASC`.
- An empty table returns no rows, with the stated column names and types.
- Do not change the input table.

## Examples

```text
user_id  event_at
1        2026-01-03 09:00:00
1        2026-01-20 18:30:00
1        2026-03-07 08:00:00
1        2026-04-02 12:00:00
2        2026-01-09 10:00:00
2        2026-01-09 11:00:00
3        2026-02-01 07:45:00
3        2026-03-05 16:00:00
4        2026-02-27 21:10:00
=> (2026-01-01, 0, 2, 2, 1.0)
   (2026-01-01, 1, 0, 2, 0.0)
   (2026-01-01, 2, 1, 2, 0.5)
   (2026-01-01, 3, 1, 2, 0.5)
   (2026-02-01, 0, 2, 2, 1.0)
   (2026-02-01, 1, 1, 2, 0.5)
   (2026-02-01, 2, 0, 2, 0.0)
```

User 2 has two events in January. The January period-0 count is 2, not 3, because the count is of
users. Users 1 and 2 form the January cohort, and user 1 is absent in February but active again in
March and April, so February shows 0 and the later periods still count user 1.

```text
user_id  event_at
9        2026-01-05 00:00:00
=> (2026-01-01, 0, 1, 1, 1.0)
```

A retried write is one user, not two. User 5's first event is logged twice:

```text
user_id  event_at
5        2026-05-02 08:00:00
5        2026-05-02 08:00:00
=> (2026-05-01, 0, 1, 1, 1.0)
```

## Constraints

- Allowed: a read-only DuckDB SQL query. CTEs, joins, window functions, `generate_series`, `UNNEST`
  and date functions are allowed.
- Not allowed: writing to or altering the `activity` table.
- Not allowed: a result that drops empty periods, or that divides by anything other than the cohort size.

## Done looks like

The result matches an independent oracle that works in calendar-month indices, on seeded random
activity with many repeated events per user and month, on a 1,000-day log with a heavy user and
retried duplicate rows (a Zipf-skewed synthetic log), and on a fixture. Hidden cases cover: a
cohort whose only period is 0; users who return after a gap; a cohort month with no new users; a
user whose events fall on the last day and first day of two months; many events by one user in one
month; a grid that extends to the latest month for an early cohort; empty input; exact output types
and ordering; a rate that is not 1.0 or 0.0; and a retried duplicate row of a first event, which
counts once in the cohort size. A query that counts events, uses the previous
period as denominator, or drops empty periods fails with the mistake named.

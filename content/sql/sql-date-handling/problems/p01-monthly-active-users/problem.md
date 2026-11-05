# Monthly active users, including months with no activity

Write one DuckDB SQL `SELECT` statement, in `submission.sql`, that returns one row for every calendar
month from November 2023 to February 2024: how many distinct users were active in that month. A month
with no activity still gets a row, with `0`. Each `month` is a DATE, the first day of the month, and
the rows come in month order.

## Contract

- Input: one table `events`.
  - `user_id` (BIGINT): the user who did something. Never NULL.
  - `event_ts` (TIMESTAMP): when it happened. Naive, with no time zone. Never NULL.
  - One row per event. A user can have many events in a month.
- Range: the four calendar months 2023-11, 2023-12, 2024-01 and 2024-02. The statement fixes the
  range; the data does not.
- An event counts toward a month when `event_ts` is at or after the first instant of that month and
  before the first instant of the next month. An event at `2024-02-01 00:00:00` belongs to February.
  An event at `2024-01-31 23:59:59` belongs to January.
- Returns: one row per month of the range, with exactly these columns in this order:
  - `month`: DATE, the first day of the month. A TIMESTAMP is not accepted.
  - `active_users`: BIGINT, the number of distinct `user_id` values with at least one event in that
    month. `0` when there are none.
- Events outside the four months are ignored.
- Row order: `month` ascending. Row order is part of the answer.
- If `events` is empty, return the four rows with `0`.
- Must read from `events`. Do not write the answer as literal rows.

## Examples

On `fixtures/events.csv`, the answer is:

```text
month       active_users
2023-11-01  3
2023-12-01  0
2024-01-01  3
2024-02-01  4
```

A small case: user 7 has two events on 2024-01-15 and one at 2024-02-01 00:00:00.

```text
user_id  event_ts
7        2024-01-15 10:00:00
7        2024-01-15 18:00:00
7        2024-02-01 00:00:00

month       active_users
2023-11-01  0
2023-12-01  0
2024-01-01  1
2024-02-01  1
```

## Constraints

- Allowed: `SELECT`, `FROM`, `JOIN`, `WHERE`, `GROUP BY`, `ORDER BY`, `WITH`, `generate_series`, and
  DuckDB's date functions and built-in aggregates.
- Not allowed: literal `VALUES` rows or any other hard-coded answer.

## Done looks like

All public tests pass, then **Submit** runs the hidden tests. They build their own `events` tables, so a
query tuned to the fixture fails. The hidden tests cover these families: an empty month in the middle, an
empty last month, an events table with no rows, events at midnight on the first of a month and at the last
second of a month, a leap-day event, several events by one user in one month (counted once), a user
active in every month, events before and after the range, the month column type, the row order, and
random tables checked against an independent pandas computation.

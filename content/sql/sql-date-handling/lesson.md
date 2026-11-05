---
title: SQL date handling
minutes: 12
min_viable: "Read The system and the Worked example, then solve p01 with the calendar hint."
---

## The system

Every date bug in SQL comes from three choices. Make them in this order.

1. **Pick the type.** `DATE` is a day. `TIMESTAMP` is a day and a time. The timestamps here are naive, with no
   time zone, so you never convert zones.
2. **Describe a period as a half-open interval.** A month is $[s, e)$, where $s$ is the first instant of the
   month and $e$ is the first instant of the next month. An event $t$ belongs to the month when $s \le t < e$.
3. **Know what each subtraction returns.** Two `DATE` values subtract to a count of days. Two `TIMESTAMP`
   values subtract to an interval of elapsed time.

Verified on locked DuckDB 2.0.0; see its [date functions](https://duckdb.org/docs/2.0/sql/functions/date) and [timestamp functions](https://duckdb.org/docs/2.0/sql/functions/timestamp):

| Expression | Result type | Meaning |
|---|---|---|
| `DATE - DATE` | `BIGINT` | calendar days |
| `TIMESTAMP - TIMESTAMP` | `INTERVAL` | elapsed time |
| `date_trunc('month', DATE)` | `TIMESTAMP` | first instant of the month, not a `DATE` |
| `generate_series(DATE, DATE, INTERVAL 1 MONTH)` | `TIMESTAMP` | includes the stop value |
| `range(DATE, DATE, INTERVAL 1 MONTH)` | `TIMESTAMP` | excludes the stop value |

## Worked example

Count distinct users per month for January and February 2024. The events are: user 1 at
`2024-01-31 23:59:59`, user 2 at `2024-02-01 00:00:00`, and user 3 at `2024-02-01 00:00:00`.

Step 1, the calendar: `2024-01-01` and `2024-02-01`, typed as `DATE`.

Step 2, the half-open test for January: $2024\text{-}01\text{-}01 \le t < 2024\text{-}02\text{-}01$.

| event | in January? | in February? |
|---|---|---|
| user 1, 2024-01-31 23:59:59 | yes | no |
| user 2, 2024-02-01 00:00:00 | no ($t = e$) | yes |
| user 3, 2024-02-01 00:00:00 | no | yes |

Result: January has 1 active user, February has 2.

Now the wrong version. `event_ts BETWEEN 2024-01-01 AND 2024-02-01` keeps the upper bound, so users 2 and 3
fall into January as well. January then reports 3, which is wrong.

## Why it works

A half-open interval partitions the timeline. The next month starts exactly where this one ends, so every
instant belongs to exactly one month. A closed interval $[s, e]$ gives its last instant to two periods. The
calendar table is the standard device for this, and Kimball and Ross build their date dimension on it
[Kimball, *The Data Warehouse Toolkit*, date dimension]. A calendar table is also the gap filler: a `LEFT JOIN`
from the calendar keeps every period, and a count of matched users returns 0 for an empty period. Molinaro's
date arithmetic chapter covers the interval forms [Molinaro, *SQL Cookbook*, date arithmetic].

## Signal vs. noise

- **Signal:** the half-open test, the calendar as the left side of the join, and the result type of each
  subtraction.
- **Signal:** `COUNT(DISTINCT user_id)`. A user with five events in a month counts once.
- **Noise for now:** time zones (not needed with naive timestamps), and the exact list of `date_part` fields.

## Where it breaks

- **`BETWEEN` on a timestamp.** The upper bound is included, so an event at the next period's first instant is
  counted twice.
- **`< last_day(month)` on a `DATE`.** Every event on the last day of the month is dropped. Use the first day of
  the next month as the exclusive bound.
- **`TIMESTAMP` compared with a `DATE`.** The `DATE` becomes midnight. `TIMESTAMP '2024-01-31 10:00'` is
  `FALSE` for `< DATE '2024-01-31'`, and `BETWEEN DATE '2024-01-01' AND DATE '2024-01-31'` excludes the 31st at
  10:00.
- **`range()` instead of `generate_series()`.** `range` stops before the stop value, so the last month disappears.
- **`date_trunc` on a `DATE` returns a `TIMESTAMP`.** Cast it back with `CAST(... AS DATE)` when the output must
  be a date.
- **Days from an interval.** `date_part('day', ts2 - ts1)` reads the day field of the elapsed interval. For
  23:59 to 00:01 on the next day it gives 0, but the calendar distance is 1. Subtract the `DATE` casts instead.
- **Weeks.** `extract('week', date)` is the ISO week. `2024-12-30` is week 1 of ISO year 2025. `dayofweek` counts
  Sunday as 0, so 2021-01-03 gives 0 and `isodow` gives 7.
- **Month arithmetic clamps.** `2023-01-31 + INTERVAL 1 MONTH` is `2023-02-28`, and `2024-01-31` gives
  `2024-02-29`.

## Interview angle

*Answer shape:* "I start from the calendar of periods, one row per month, typed as dates. I LEFT JOIN the events
with a half-open test, so an empty month keeps its row. I count distinct users, then cast the result back to a
date." Mention the boundary rule before writing the join.

*Follow-up 1: time zones?* Convert every timestamp to one zone before truncating. Here the timestamps are naive,
so the question does not arise.

*Follow-up 2: why not count the events directly?* A direct `GROUP BY` lists only the months that have events.
An empty month needs the calendar.

## Connections

- **Prerequisites:** [sql/sql-subqueries-and-ctes] for naming the calendar in a CTE, and
  [sql/sql-joins-and-unions] for the grain and the LEFT JOIN.
- **Unlocks:** cohort and retention queries, which are calendar problems with a second date.
- **Lineage:** the calendar table is standard in analytics warehouses, and Kimball's date dimension is the
  reference design.

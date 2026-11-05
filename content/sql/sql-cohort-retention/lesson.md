---
title: "SQL cohort retention"
minutes: 12
min_viable: "Read The system and the Worked example, then solve p01 by writing the four CTEs in order."
---

## The system

A **cohort retention table** asks: of the users who started in a given month, how many were active
in each later month?

Four definitions, stated before any SQL:

| Symbol | Definition | Example |
|---|---|---|
| $m(e)$ | Activity month of event $e$: the first day of its calendar month | `2026-03-07` gives `2026-03-01` |
| $c_u$ | Cohort of user $u$: the earliest activity month of $u$ | the month of the first event |
| $p$ | Period offset: whole calendar months from the cohort month | $p = 0$ is the cohort's own month |
| $A(c, p)$ | Active users: distinct users with $c_u = c$ and an event in month $c + p$ | a set size, not a row count |

The cohort size is $N_c = |\{u : c_u = c\}|$, and the retention rate is

$$R(c, p) = \frac{A(c, p)}{N_c}, \qquad 0 \le R(c, p) \le 1.$$

The denominator is always $N_c$, never the previous period's count.

**The grid.** For cohort $c$, periods run from $0$ to $L - c$, where $L$ is the latest activity
month in the whole table. A period with no active user is still a row, with $A = 0$ and $R = 0$.

**The output contract** (full text in p01). Columns: `cohort_month` DATE, `period` BIGINT,
`active_users` BIGINT, `cohort_size` BIGINT, `retention_rate` DOUBLE = `ROUND(active_users /
cohort_size, 4)`. Rows: each cohort with a user, times periods 0 to $L - c$, ordered by cohort then
period. An empty table returns no rows.

## Worked example

Four users, all in 2026. Each row is one event.

| user | events (month of each event) |
|---|---|
| 1 | Jan 3, Jan 20, Mar 7, Apr 2 |
| 2 | Jan 9, Jan 9 |
| 3 | Feb 1, Mar 5 |
| 4 | Feb 27 |

The latest month is $L = $ April. Cohorts: Jan (users 1, 2) and Feb (users 3, 4). The grid is 7
rows: Jan has periods 0 to 3, Feb has periods 0 to 2.

Cohort Jan, $N = 2$:

| $p$ | month | distinct active users | $R$ |
|---|---|---|---|
| 0 | Jan | {1, 2} = 2 | 2/2 = 1.0 |
| 1 | Feb | {} = 0 | 0/2 = 0.0 |
| 2 | Mar | {1} = 1 | 1/2 = 0.5 |
| 3 | Apr | {1} = 1 | 1/2 = 0.5 |

User 1 is absent in February and returns in March, so the February row stays at 0. User 2 has
two January events. Counting events would give 4 at period 0 and a rate of 2.0. Counting distinct
users gives 2 and 1.0.

Cohort Feb, $N = 2$: period 0 gives 2 (1.0), period 1 gives {3} = 1 (0.5), period 2 gives 0 (0.0).

## Why it works

Each user has one cohort, so cohort sizes partition the users. The join only relabels events; the
aggregation carries the meaning. `COUNT(DISTINCT user_id)` gives $A(c, p)$. `COUNT(*)` counts events.

The grid comes from a calendar, not the data: a cross join with `generate_series(0, L - c)` makes
every cell exist. A `LEFT JOIN` plus `COALESCE(..., 0)` fills the gaps. An inner join would drop
the zeros and flatter the curve.

## Signal vs. noise

- **Signal:** the cohort definition, the denominator, the distinct count, the grid, and month
  arithmetic on truncated dates.
- **Noise for now:** weekly cohorts and survival-curve methods such as Kaplan-Meier.

## Where it breaks

- **Counting events.** Several events in one month inflate the cell. Use `COUNT(DISTINCT user_id)`.
- **Denominator from the previous period.** Dividing by the period $p - 1$ count turns the table
  into a chain of conversion rates. Keep $N_c$.
- **Denominator from the active users.** Dividing by the active count makes every cell 1.0.
- **Inner join on activity.** Empty cells disappear, so a period with no return looks missing
  instead of zero.
- **Day arithmetic.** `DATE_DIFF('day', ...) / 30` is a 30-day bucket, not a calendar month. From
  1 Feb to 1 Mar 2026 is 28 days, so `/30` gives 0 where the answer is period 1. Over four years
  the drift shows: 1 Jan 2020 to 1 Jan 2024 is 1,461 days, and $1461 / 30 = 48.7$ rounds to 49,
  but the answer is 48. Short cases pass this mistake; only the
  four-year hidden test catches it. Use `DATE_DIFF('month', ...)` on truncated dates.
- **Rounding.** Round once, at the end, to 4 decimal places. The grader allows half a unit in the
  fourth decimal ($5 \times 10^{-5}$), because ties can round either way. It also requires a
  multiple of $10^{-4}$, so an unrounded rate fails.

## Interview angle

State the four definitions, then the four CTEs. Name the two traps first: events versus users, and
the denominator. Full answer in `interview.md`.

## Connections

`sql/sql-date-handling` gives month truncation and the calendar grid. `sql/sql-group-by-and-having`
gives the grouping and distinct count. Pivoting the table into a matrix is a display step.

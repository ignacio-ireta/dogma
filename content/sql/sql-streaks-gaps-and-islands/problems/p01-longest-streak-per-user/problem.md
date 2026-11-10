# Longest streak of consecutive active days per user

A reading app records a row each time a user opens it. A user can open the app several times on one day,
and a day with no row is a day off. A **streak** is a run of consecutive calendar days on which the user
was active, where each next day is exactly one calendar day after the previous one. Write one query that
returns each user's longest streak, with its first and last day. If two of a user's streaks tie in length,
return the one that starts earlier.

## Contract

Table `visits` has columns:

| Column | DuckDB type | Rules |
|---|---|---|
| user_id | BIGINT | Non-null |
| visit_date | DATE | Non-null; many rows can share one user and one date |

- Duplicate rows for the same user and date count as one active day.
- A streak's length is the number of distinct active days in it. Two visits on one day do not make a longer
  streak.
- For each user with at least one visit, return one row with the longest streak. If two of a user's streaks
  have the same length, return the one with the earlier start.
- `streak_days` is the number of days in that streak. `streak_start` and `streak_end` are its first and last
  dates, inclusive.
- Output columns, in order, with these types: `user_id BIGINT`, `streak_days BIGINT`, `streak_start DATE`,
  `streak_end DATE`.
- Final output order is `user_id ASC`.
- A run continues from one calendar date to the next across month ends, year ends and leap days: 31 December 2028 is followed by 1 January 2029, and 28 February 2028 by 29 February.
- Users with no visits do not appear. An empty table returns no rows, with the stated columns and types.
- Do not change the `visits` table.

## Examples

```text
user_id visit_date
1       2026-01-01
1       2026-01-02
1       2026-01-02      <- duplicate day, counts once
1       2026-01-03
1       2026-01-05
1       2026-01-06
2       2026-03-10
3       2026-02-27
3       2026-02-28
3       2026-03-01      <- February 2026 has 28 days, so Feb 28 is followed by Mar 1
4       2026-01-01
4       2026-01-02
4       2026-01-10
4       2026-01-11
=> (1, 3, 2026-01-01, 2026-01-03)
   (2, 1, 2026-03-10, 2026-03-10)
   (3, 3, 2026-02-27, 2026-03-01)
   (4, 2, 2026-01-01, 2026-01-02)
```

User 1 has two runs: January 1 to 3 (three days) and January 5 to 6 (two days). The longest is the first.
User 4 has two runs of two days. The tie goes to the earlier start, January 1.

```text
user_id visit_date
9       2026-07-04
=> (9, 1, 2026-07-04, 2026-07-04)
```

```text
user_id visit_date
5       2028-12-29
5       2028-12-30
5       2028-12-31
5       2029-01-01
5       2029-01-02
5       2029-01-03
=> (5, 6, 2028-12-29, 2029-01-03)
```

A run that crosses 31 December 2028 is one streak. The year end does not restart the count.

## Constraints

- Allowed: a read-only DuckDB SQL query. CTEs, subqueries, window functions, `DISTINCT`, `GROUP BY` and
  `QUALIFY` are allowed.
- Not allowed: writing to or altering the `visits` table.
- The grader checks results only. Any query that returns the rows above is accepted.

## Done looks like

The result matches an independent oracle that takes each user's distinct dates, sorts them, and walks the
runs in Python. The seeded fixture and the hidden cases cover: duplicate visits on one day; a single-day
user; a gap of exactly one day that breaks a run; a run across a month end and a year end; a leap day; ties
in length, where the earliest start must win; a user whose longest run comes after a shorter one; many users
in shuffled insertion order; an empty table; the exact output types; and the final ordering. A dense holiday burst across a leap-year end, with heavy-tailed users and duplicate visits, is the realistic family.

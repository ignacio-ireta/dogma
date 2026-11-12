# All streaks of consecutive active days per user

The same reading-app `visits` table as the longest-streak problem. A **streak** (island) is a maximal run of
consecutive calendar days on which a user was active. Two visits on one day are one active day, and a day
with no visit ends the run. Write one query that returns every streak of every user.

## Contract

Table `visits` has columns:

| Column | DuckDB type | Rules |
|---|---|---|
| user_id | BIGINT | Non-null |
| visit_date | DATE | Non-null; many rows can share one user and one date |

- Duplicate rows for the same user and date count as one active day.
- Each streak is a maximal run: its first day has no active previous day, and its last day has no active
  next day. A single active day is a streak of length one.
- Output columns, in order, with these types: `user_id BIGINT`, `island_start DATE`, `island_end DATE`,
  `island_days BIGINT`. `island_start` and `island_end` are inclusive and `island_days` counts the active days.
- Final output order is `user_id ASC, island_start ASC`.
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
=> (1, 2026-01-01, 2026-01-03, 3)
   (1, 2026-01-05, 2026-01-06, 2)
   (2, 2026-03-10, 2026-03-10, 1)
```

```text
user_id visit_date
4       2026-01-01
4       2026-01-02
4       2026-01-10
4       2026-01-11
=> (4, 2026-01-01, 2026-01-02, 2)
   (4, 2026-01-10, 2026-01-11, 2)
```

```text
user_id visit_date
5       2028-12-30
5       2028-12-31
5       2029-01-01
=> (5, 2028-12-30, 2029-01-01, 3)
```

## Constraints

- Allowed: a read-only DuckDB SQL query. CTEs, subqueries, window functions, `DISTINCT`, `GROUP BY` and
  `QUALIFY` are allowed.
- Not allowed: writing to or altering the `visits` table.
- The grader checks results only. Any query that returns the rows above is accepted.

## Done looks like

The result matches an independent oracle that takes each user's distinct dates, sorts them, and splits them
at every gap of more than one day. The seeded fixture and the hidden cases cover: one streak per user; many
streaks per user; duplicate visits; single-day streaks; a gap of exactly one day; a run across a leap day and a leap-year end; real retail activity (`online_retail`) moved onto a leap-year end, with heavy-tailed users and duplicate visits;
consecutive days of two different users, which must not join; users with no visits absent; an empty table; the exact output types; and the final ordering.

The hidden tests draw their bulk inputs from the real dataset `online_retail` (see
`docs/DATASETS.md`).

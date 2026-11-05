## Hint 1 — concept

A retention table answers one question per cell: of the users who started in a given month, how
many were active in a later month? Before you write any SQL, decide three things: what the cohort
is, what the period is, and what the denominator is. Then ask what a count of rows measures
compared with a count of people.

## Hint 2 — formula / approach

Truncate every event to the first day of its month. Each user's cohort is the minimum of those
dates. The period is the number of calendar months between the cohort month and an activity
month. The active count is the number of distinct users in each cohort and period. The denominator
is the number of users in the cohort, not the active count of the previous period. Finally, a
period with no active users must still appear in the output, so build the grid of periods first
and join the counts onto it.

## Hint 3 — code shape

Use a CTE for each step, in this order: monthly events, first month per user, cohort sizes, active
counts grouped by cohort and period, and the grid. For the grid, a set-returning `generate_series`
inside `UNNEST` gives one row per period. Finish with a `LEFT JOIN` to the counts, a `COALESCE` on
the active count, and a `ROUND` on the rate.

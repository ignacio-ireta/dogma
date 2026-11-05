# Interview notes: SQL conditional aggregation

## Q1. How do you compute several conditional counts and sums per group in one query?

**60-second answer.** Group once, then put each condition inside its own aggregate. A count of
matches is `COUNT(*) FILTER (WHERE ...)`, or a `SUM` of a `CASE` that returns 1 or 0. A sum of a
value over matches is `SUM(CASE WHEN ... THEN value END)`. The query scans the table once, and each
column reads the same groups. The sum needs `COALESCE` when a group can have no matching non-NULL value,
because the sum is then NULL, not 0.

**Follow-ups.**
- *Why not one query per condition?* Each query scans the table again, and the results then need a
  join to sit side by side. Conditional aggregation avoids both.
- *Is `FILTER` standard?* It is standard SQL and supported by DuckDB, which this platform uses. The
  `CASE` form is the portable fallback.

## Q2. How do you compute a conversion rate per group, and what can go wrong?

**60-second answer.** The rate is the matches over the group size: a count of converted rows divided
by `COUNT(*)`. The division must be DOUBLE, because two integers in an integer quotient truncate
to 0. Guard the zero denominator with `NULLIF`, round to a stated number of places, and make the zero
case explicit: a group with no conversions shows 0.0, not NULL. I check the unit too: a fraction,
not a percentage.

**Follow-ups.**
- *What does `COUNT(converted)` return?* The row count, because FALSE is not NULL. It is the most
  common silent error in conversion queries.
- *Why not `WHERE converted`?* It removes the other sessions before grouping, so the denominator
  shrinks and channels with no conversions disappear.

## Common mistakes
- Filtering the flag in `WHERE` when the question needs both the total and the matches.
- `COUNT(flag)` used as a count of TRUE values.
- Integer division (`//`, or a cast of the quotient to INTEGER) for a rate.
- A conditional `SUM` with no `COALESCE`, so an empty group shows NULL.
- A rate that is NULL for a group with zero conversions, when the answer is 0.0.

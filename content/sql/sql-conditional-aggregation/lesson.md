---
title: SQL conditional aggregation
minutes: 12
min_viable: "Read The system and the Worked example, then solve p01."
---

## The system

A group-by answers one question per group. Conditional aggregation answers several questions about
the same groups in one pass, because each aggregate carries its own condition.

Three forms count "the rows where X":

| Form | Counts | Note |
|---|---|---|
| `COUNT(*) FILTER (WHERE X)` | rows where X is TRUE | Standard SQL, supported by DuckDB. Clearest. |
| `COUNT(CASE WHEN X THEN 1 END)` | rows where X is TRUE | CASE gives NULL for the rest, and COUNT skips NULL. |
| `SUM(CASE WHEN X THEN 1 ELSE 0 END)` | rows where X is TRUE | Gives 0 for a group with no match. |

Choose the aggregate by what it counts. For a `converted` flag holding TRUE, FALSE, FALSE:

| Expression | Result | Why |
|---|---|---|
| `COUNT(*)` | 3 | every row |
| `COUNT(converted)` | 3 | non-NULL values, and FALSE is non-NULL |
| `COUNT(*) FILTER (WHERE converted)` | 1 | matches only |

A conditional sum has one more rule. `SUM(CASE WHEN device = 'mobile' THEN revenue END)` skips the
other rows, but when no mobile row exists the result is NULL, not 0.

## Worked example

Six sessions, two channels. Per channel: sessions, conversions, rate, mobile and desktop revenue.

| session | channel | device | converted | revenue |
|---|---|---|---|---|
| 1 | ads | mobile | TRUE | 40.0 |
| 2 | ads | desktop | FALSE | NULL |
| 3 | ads | mobile | FALSE | NULL |
| 4 | ads | desktop | TRUE | NULL |
| 5 | ads | mobile | TRUE | 25.0 |
| 6 | email | desktop | FALSE | NULL |

1. **Group by channel.** Two bags: ads = {1 to 5}, email = {6}.
2. **Sessions.** `COUNT(*)`: ads 5, email 1.
3. **Conversions.** `SUM(CASE WHEN converted THEN 1 ELSE 0 END)`: sessions 1, 4 and 5 for ads, so 3. Email 0.
4. **Rate.** `CAST(3 AS DOUBLE) / 5 = 0.6` for ads. Email gives 0.0. `NULLIF(COUNT(*), 0)` guards the divide.
5. **Revenue.** `COALESCE(SUM(CASE WHEN device = 'mobile' THEN revenue END), 0.0)` gives 65.0 for ads
   (40.0 + 25.0) and 0.0 for email, which has no mobile row. Ads has only NULL desktop revenue, so 0.0.

| channel | sessions | converted_sessions | conversion_rate | mobile_revenue | desktop_revenue |
|---|---|---|---|---|---|
| ads | 5 | 3 | 0.6 | 65.0 | 0.0 |
| email | 1 | 0 | 0.0 | 0.0 | 0.0 |

## Why it works

An aggregate reduces a bag of rows to one value, and a `CASE` or `FILTER` decides which rows enter it.
The grouping runs once, so every column reads the same bags, with no join.

SQL aggregates skip NULL (Beaulieu, *Learning SQL*, the material on grouping and aggregates). A `CASE`
without an `ELSE` returns NULL for the rows it does not match, so `SUM` and `COUNT` skip them. That is
why `CASE WHEN X THEN v END` sums `v` where X holds, and why that sum needs `COALESCE` when no row matches.

Division has its own rule. In DuckDB, `/` on two integers returns a DOUBLE: `1 / 2` is `0.5`. The
integer quotient is `//`, and `1 // 2` is `0`. Casting the numerator to DOUBLE keeps the intent visible
on any engine. A zero denominator is not an error in DuckDB: `1.0 / 0.0` is `inf`, and `0.0 / 0.0` is NaN.
`NULLIF(n, 0)` turns the zero into NULL.

## Signal vs. noise

- **Signal:** the difference between `COUNT(*)`, `COUNT(column)` and a sum of a 0/1 flag. The NULL
  rule for a sum over `CASE`, and its `COALESCE`. Division as DOUBLE, guarded by `NULLIF`.
- **Signal:** each conditional aggregate is one output column. A new condition adds a column, not a query.
- **Noise for now:** engine-specific `PIVOT` syntax. The `CASE` form works everywhere.

## Where it breaks

- **A WHERE on the flag.** `WHERE converted` before grouping removes the other rows from every
  aggregate. Totals shrink, and a channel with no conversions vanishes. Put the condition inside the aggregate.
- **`COUNT(flag)`.** It counts TRUE and FALSE alike. Use `COUNT(*) FILTER` or a `CASE` that returns NULL.
- **Integer quotients.** `SUM(...) // COUNT(*)` gives 0 for a rate of 0.6. Divide as DOUBLE.
- **Filtered denominators.** `COUNT(*) FILTER (WHERE device = 'mobile')` is 0 for a channel with no
  mobile session, and dividing by it gives `inf` unless it is guarded with `NULLIF`.

Dirty event_log channel names create 30 exact groups from six source labels. For the ads label,
there are 741 exact sessions; trimming and lowercasing merges in 121 more. Revenue also has a long
tail: the 99th percentile in this draw is 281.29.

## Interview angle

"Conversion rate and mobile revenue by channel, in one query." Answer shape: `GROUP BY channel`, one
aggregate per column, each with its condition inside it. `COUNT(*) FILTER (WHERE converted)` for the
conversions, `COALESCE(SUM(CASE ...), 0)` for each revenue split, and a DOUBLE division guarded by
`NULLIF` for the rate. Follow-ups:

- *Why not `WHERE converted` for the conversions?* It deletes the non-converting sessions, so the
  total is wrong and channels with no conversions vanish.
- *How would you check it?* Use a channel with no conversions, one with a single device, and one with
  NULL revenue, and work out the expected rows by hand.

## Connections

- **Requires:** `sql/sql-group-by-and-having`, for grouping and the NULL rules.
- **Unlocks:** reporting tables with several measures, and the pandas version of the same reshape,
  in `sql/pandas-merge-and-reshape`.
- **Reference:** Molinaro, *SQL Cookbook*, the reporting and pivot recipes.

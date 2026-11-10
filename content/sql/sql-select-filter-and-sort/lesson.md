---
title: SQL select, filter and sort
minutes: 12
min_viable: "Read The system and the Worked example, then solve p01."
---

## The system

A `SELECT` is a pipeline over a table. You write the clauses in one order, but the engine
evaluates them in another:

| Step | Clause | What it does | What it can see |
|---|---|---|---|
| 1 | `FROM` | Names the table | the stored rows |
| 2 | `WHERE` | Keeps rows whose condition is TRUE | the rows from `FROM`; not output aliases (standard SQL) |
| 3 | `SELECT` | Computes the output columns | the rows that survived `WHERE` |
| 4 | `ORDER BY` | Sorts the output | output columns, including aliases |
| 5 | `LIMIT` | Keeps the first $n$ rows | the sorted output |

Three rules follow from the pipeline:

1. **A row survives `WHERE` only if the condition is TRUE.** FALSE and UNKNOWN both drop it.
2. **A table has no row order.** Only `ORDER BY` defines one. `LIMIT` without `ORDER BY` returns
   an arbitrary set of rows.
3. **Ties in `ORDER BY` are unspecified.** Add a unique key as the last sort key.

Filter decision list for a condition on one column:

- **Exact value:** `status = 'completed'`. Avoid `status <> 'pending'`. It keeps `refunded`,
  `processing` and every status you have not thought of yet.
- **Date range:** `order_date >= DATE '2024-01-01' AND order_date < DATE '2025-01-01'`. The upper
  bound is open, so it works for timestamps too.
- **Missing value:** `amount IS NULL` or `amount IS NOT NULL`. Never `amount = NULL`.

## Worked example

Table `orders` (6 rows):

| order_id | status | order_date | amount |
|---|---|---|---|
| 1 | completed | 2024-01-05 | 40.00 |
| 2 | pending | 2024-02-02 | 310.00 |
| 3 | completed | 2024-03-21 | 250.00 |
| 4 | completed | 2024-04-09 | 250.00 |
| 5 | completed | 2023-12-31 | 500.00 |
| 6 | completed | 2024-05-01 | NULL |

Question: the single largest completed order of 2024.

1. **FROM:** six rows.
2. **WHERE** `status = 'completed' AND order_date >= DATE '2024-01-01' AND order_date < DATE '2025-01-01' AND amount IS NOT NULL`.
   Row 2 fails on status. Row 5 fails on date. Row 6 fails the NULL check. Rows 1, 3 and 4 remain.
3. **SELECT** `order_id, amount` for those three rows.
4. **ORDER BY** `amount DESC, order_id ASC`: rows 3 and 4 tie at 250.00, so order 3 comes first.
   Then row 1.
5. **LIMIT 1:** order 3.

The tie-breaker decides the answer. Without `order_id ASC`, the engine may return order 4, and
nothing in the query says which is right.

Why the NULL test is needed. A condition on a NULL value is UNKNOWN, not FALSE:

| Expression | Result when `amount` is NULL | Row kept by `WHERE`? |
|---|---|---|
| `amount > 100` | UNKNOWN | no |
| `amount = NULL` | UNKNOWN (never TRUE) | no |
| `NOT (amount > 100)` | UNKNOWN | no |
| `amount IS NULL` | TRUE | yes |

## Why it works

SQL is a set language. A query names a result set and the pipeline defines it, so the engine
is free to run the steps in any physical order that gives the same answer. The one guaranteed
order is the one you write with `ORDER BY`.

Comparisons with NULL return UNKNOWN, a third truth value, and `WHERE` keeps only TRUE. This is
why `NOT (amount > 100)` does not return the NULL rows: they are UNKNOWN on both sides. Missing
values fall out of every condition unless you test for them explicitly (Beaulieu, *Learning SQL*, chapter on filtering).

A half-open range `>= start AND < next_start` covers exactly one calendar year whether the column
is a DATE or a TIMESTAMP. `BETWEEN '2024-01-01' AND '2024-12-31'` misses 2024-12-31 at 09:00 when
the column is a timestamp, because the bound is read as midnight.

## Signal vs. noise

- **Signal:** the five-step pipeline. `WHERE` decides which rows are in. `ORDER BY` is the only
  source of order. A comparison with NULL is UNKNOWN. A deterministic answer needs a tie-breaker.
- **Signal:** the half-open date range, written the same way every time.
- **Noise for now:** dialect date helpers (`DATE_TRUNC`, `EXTRACT` variants), `FETCH FIRST n ROWS`
  versus `LIMIT`, and `QUALIFY`. You will meet them, but the logic does not change.

## Where it breaks

- **`LIMIT` without `ORDER BY`.** You get five arbitrary rows. The result can change between runs
  or between engines.
- **`x NOT IN (...)` with a NULL in the list.** The result is UNKNOWN for every row that is not
  matched, so the query returns no rows. This is the most common NULL trap in production SQL.
- **Output aliases in `WHERE`.** Standard SQL rejects `WHERE revenue > 100` when `revenue` is an
  alias. DuckDB accepts it as an extension. Do not rely on it when you move to Postgres or SQL
  Server.
- **`<>` on a nullable column.** `status <> 'pending'` drops rows where `status` is NULL, which
  is usually not what you meant.
- **Timestamp upper bounds.** `< '2024-12-31'` on a timestamp drops that whole last day. Use
  `< '2025-01-01'`.

The event_log source has lognormal order amounts and dirty statuses. In a seeded 5,000-row draw (200
users, `dirty_rate` 0.15), the largest 2024 order with a dirty completed label is 311.28, stored as
`'completed '`. It is the 12th completed-like order by amount. Exact matching drops it, so the exact
12th row is 307.74, not 311.28; the top five are the same either way.

## Interview angle

"Write the query for the five largest orders last year." Answer shape: name the three filters
(status, period, NULL amount), use a half-open date range, order by value descending with a
unique tie-breaker, and apply `LIMIT`. Follow-ups:

- *Two orders tie at the fifth place.* The tie-breaker decides which one is in. Say so, and say
  which key you chose and why.
- *How do you treat NULL amounts?* Decide, then write it down. Excluding them is a business rule,
  not a default. DuckDB sorts NULLs last, so they only appear if fewer than five orders qualify.

## Connections

- **Unlocks:** `sql/sql-group-by-and-having` (the same pipeline with grouping added),
  `sql/sql-top-n-and-latest-row-per-group` (top N inside each group), and
  `sql/sql-ranking-window-functions`.
- **Lineage:** Codd's relational model (1970) makes a query a set-valued expression. SQL, built on
  that model, was standardised in 1986. Three-valued logic is part of the standard, so the NULL
  behaviour here is the same in most engines.

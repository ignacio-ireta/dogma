---
title: SQL subqueries and CTEs
minutes: 12
min_viable: "Read The system and the Worked example, then solve p01."
---

## The system

A subquery is a `SELECT` used inside another query. The shape of the answer it returns decides
which form you need:

| The inner query returns... | Form | The outer query uses it as... |
|---|---|---|
| one value | scalar subquery | a number: `WHERE amount > (SELECT AVG(amount) FROM orders)` |
| one column, many rows | `IN` subquery | a set: `WHERE customer_id IN (SELECT customer_id FROM vip)` |
| rows, or nothing | `EXISTS` subquery | a test: `WHERE EXISTS (SELECT 1 FROM refunds r WHERE r.order_id = o.order_id)` |
| a table | derived table | a source: `FROM (SELECT ...) AS t` |
| one value per outer row | correlated subquery | a per-row number: it refers to a column of the outer row |

A correlated subquery is logically evaluated once per outer row. An engine may rewrite it as a
join. That is the engine's choice, so judge the query by its result.

A **common table expression** (CTE) names a subquery at the top of the query:

```sql
WITH buyer_totals AS (
    SELECT customer_id, SUM(amount) AS total_spend
    FROM orders WHERE status = 'completed' GROUP BY customer_id
)
SELECT ... FROM buyer_totals ...
```

A later CTE can read an earlier one.

Decision rule: when a comparison needs a number that summarises many rows, name that number in a
CTE. Use a correlated subquery only when the number depends on the row, such as its own
customer's average.

## Worked example

Question: which customers have a total above the average buyer total? Five orders:

| order_id | customer | status | amount |
|---|---|---|---|
| 1 | A | completed | 40 |
| 2 | A | completed | NULL |
| 3 | A | completed | 10 |
| 4 | B | completed | 20 |
| 5 | C | pending | 99 |

1. **Stage 1, `buyer_totals`.** Keep completed rows and group by customer. A's total is
   $40 + \text{NULL} + 10 = 50$, because `SUM` skips the NULL. B's is 20. C has no completed row,
   so C is not a buyer.
2. **Stage 2, `average_total`.** `AVG` over the buyer totals: $(50 + 20) / 2 = 35$.
3. **Final.** Keep totals strictly above 35. A (50) is returned. B (20) is not.

## Why it works

A scalar subquery must return one value. Two rows is a run-time error, and zero rows gives NULL.

NULL is where subqueries go wrong. `x NOT IN (1, NULL)` means `x <> 1 AND x <> NULL`. The second
term is NULL for every `x`, so the test is never TRUE and `WHERE` keeps no row. A `NOT IN` over a
subquery that returns a NULL therefore returns nothing. `x IN (...)` survives a NULL: it is TRUE on
a match and NULL otherwise. `NOT EXISTS` asks whether a matching row exists, and a NULL matches
nothing. That is three-valued logic (TRUE, FALSE, NULL) at work.

## Signal vs. noise

- **Signal:** the decision table, and which form returns what. The `NOT IN` NULL trap. A scalar
  subquery returns one value.
- **Signal:** the population of an average. Decide who is in it (buyers, not every customer) and how
  NULL behaves (`AVG` skips it). Most wrong answers come from here.
- **Noise for now:** plan details such as decorrelation or `MATERIALIZED` CTE hints.

## Where it breaks

- **`NOT IN` on a column that can be NULL.** Use `NOT EXISTS`, or filter the subquery with
  `WHERE col IS NOT NULL`.
- **A scalar subquery that returns several rows.** DuckDB raises an error. Aggregate it, or make it
  return one row.
- **The wrong population.** Averaging order amounts when the question is about customers, or
  counting customers with no orders as 0.
- **NULL keys.** A `GROUP BY` on a column with NULLs makes a NULL group, and a join drops it.
- **Orphan keys.** An order whose `customer_id` has no row in `customers` is not a customer. If the
  average is taken before an inner join to `customers`, that order's total still enters the average.
  Join first, then aggregate, so the population of the average is the population of the question.
- **Window functions in `WHERE`.** Not allowed. Compute them in a CTE and filter in the outer
  query. A `SELECT` alias in `WHERE` is not standard SQL, though DuckDB accepts it.
- **Cents in DOUBLE and a strict `>`.** Three orders of 12.34 average to 12.339999999999998 in
  DOUBLE, so `amount > AVG(amount)` calls all three above their own average. Over the 8,094
  distinct cent amounts in the vendored orders file, each repeated 2 to 11 times, DuckDB's average
  lands below the value in 4.4% of cases. `AVG` of a DECIMAL column still returns DOUBLE; the exact
  test is `amount * COUNT(*) > SUM(amount)` on DECIMAL or integer cents. The problems here use
  amounts in multiples of 0.25, which DOUBLE holds exactly.

**Preview.** `AVG(amount) OVER (PARTITION BY customer_id)` keeps every row and adds the group's
value beside it. You still filter on that value in a later query.

## Interview angle

"Which customers spend more than the average customer?" Define the population first (buyers, with at
least one completed order) and how NULLs behave. Then stage it: a CTE for the totals, a CTE for the
average, and a strict `>` in the final query. Follow-ups:

- *Why does my `NOT IN` return nothing?* The subquery contains a NULL. Use `NOT EXISTS`, or filter
  the NULLs out.
- *When would you use a window function instead?* When you keep every row and need the group's value
  beside it, such as each order next to its customer's average.

## Connections

- **Prerequisites:** `sql/sql-group-by-and-having` (aggregates, `WHERE` against `HAVING`) and
  `sql/sql-joins-and-unions` (the grain problem).
- **Unlocks:** `sql/sql-ranking-window-functions` and `sql/sql-running-totals-and-frames`, where a
  window function replaces a correlated subquery, and `sql/sql-top-n-and-latest-row-per-group`.
- **Lineage:** a subquery is a nested expression. The NULL rules come from three-valued logic.

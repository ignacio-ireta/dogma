---
title: SQL GROUP BY and HAVING
minutes: 12
min_viable: "Read The system and the Worked example, then solve p01."
---

## The system

Grouping adds two steps to the `SELECT` pipeline from the previous topic:

$$\text{FROM} \rightarrow \text{WHERE} \rightarrow \textbf{GROUP BY} \rightarrow \textbf{HAVING}
\rightarrow \text{SELECT} \rightarrow \text{ORDER BY} \rightarrow \text{LIMIT}$$

1. `GROUP BY` collapses the surviving rows into one bag per distinct key. Each bag is one output row.
2. Every column in `SELECT` must be either a group key or inside an aggregate. `SELECT customer_id,
   order_id ... GROUP BY customer_id` is an error, because one customer has many orders.
3. `WHERE` filters **rows** before grouping. `HAVING` filters **groups** after aggregation, so it
   may use `COUNT(*)` or `SUM(amount)`. `WHERE` may not, because the aggregates do not exist yet.

Decision rule: ask what the condition is about.

| The condition is about... | Put it in | Example |
|---|---|---|
| one order's own column | `WHERE` | `status = 'completed'` |
| the group key | `WHERE` (fewer rows to group) | `customer_id <> 0` |
| an aggregate of the group | `HAVING` | `COUNT(*) >= 3` |

How the aggregates treat NULL. Take three rows with amounts 40, NULL and 10:

| Expression | Result | Why |
|---|---|---|
| `COUNT(*)` | 3 | counts rows |
| `COUNT(amount)` | 2 | counts non-NULL amounts |
| `SUM(amount)` | 50 | skips the NULL |
| `AVG(amount)` | 25 | $50 / 2$, the non-NULL count |
| `SUM(amount) / COUNT(*)` | 16.67 | $50 / 3$, not an average |

## Worked example

Orders (8 rows). Question: which customers have at least three completed orders, and what do
those orders total?

| order_id | customer | status | amount |
|---|---|---|---|
| 1 | A | completed | 40 |
| 2 | A | completed | NULL |
| 3 | A | completed | 10 |
| 4 | A | cancelled | 99 |
| 5 | B | completed | 20 |
| 6 | B | completed | 30 |
| 7 | B | refunded | 5 |
| 8 | C | completed | 15 |

1. **WHERE** `status = 'completed'` removes rows 4 and 7. Rows 1, 2, 3, 5, 6 and 8 remain.
2. **GROUP BY customer** gives three bags: A = {1, 2, 3}, B = {5, 6}, C = {8}.
3. **Aggregates:** A has `COUNT(*) = 3` and `SUM = 50`. B has `COUNT(*) = 2` and `SUM = 50`.
   C has `COUNT(*) = 1` and `SUM = 15`.
4. **HAVING** `COUNT(*) >= 3` keeps A only.

Result: customer A, 3 completed orders, total 50.

Two traps in this example. Without the `WHERE`, customer B has three orders (row 7 is refunded)
and would wrongly qualify. If you test `COUNT(amount)` instead of `COUNT(*)`, A drops to 2 because
of the NULL.

## Why it works

A group is a bag of rows that share a key. An aggregate reduces a bag to one value. `HAVING` is a
filter on that one-row-per-group result, which is why it can see aggregates and `WHERE` cannot.
Filtering rows before grouping changes what the aggregates see, which is why `WHERE` is the place
for row conditions. Both placements can give the same result for a condition on the group key, but
`WHERE` is cheaper because fewer rows reach the grouping step.

NULL handling is a rule, not a default you can ignore. NULL means "unknown". SQL aggregates skip
NULL values (`SUM`, `AVG`, `MIN`, `MAX`, `COUNT(col)`), and `COUNT(*)` counts rows regardless.
A group with no non-NULL values gives NULL from `SUM`, not 0. Use `COALESCE(SUM(amount), 0)` only
when 0 is the right business answer (Beaulieu, *Learning SQL*, chapter on grouping and aggregates).

## Signal vs. noise

- **Signal:** the WHERE/HAVING decision rule. The completeness rule for `SELECT` columns. The
  difference between `COUNT(*)` and `COUNT(col)`. SUM and AVG skip NULLs.
- **Signal:** a tie-breaker on any sorted group result. Group output has no defined order.
- **Noise for now:** `GROUPING SETS`, `ROLLUP` and `CUBE`. `FILTER (WHERE ...)` is equivalent to
  a `CASE` inside an aggregate. You will see both, and both are fine.

## Where it breaks

- **Joins before grouping.** If a customer has several rows in a second table, joining it to
  `orders` repeats each order once per match. `COUNT(*)` and `SUM` then inflate. Decide the grain
  (one row per order) before you join.
- **AVG versus SUM over COUNT.** `AVG` divides by the non-NULL count. `SUM / COUNT(*)` divides by
  all rows. They differ as soon as one value is NULL.
- **Unordered groups.** Without `ORDER BY`, the order of groups is not defined. Tests and reports
  that compare results must sort them.
- **`HAVING` for row filters.** `HAVING status = 'completed'` either errors or filters the wrong
  thing. Row conditions belong in `WHERE`.

An event_log draw of 3,000 orders has 1,286 exact completed rows and 57 customers meeting the
three-order threshold. Lowercasing and trimming status labels changes that to 1,526 rows and 61
customers. Treat status as the exact stored key.

## Interview angle

"How many customers placed at least three orders in the last 90 days?" Answer shape: `WHERE` for
the row conditions (date, status), `GROUP BY customer_id`, `HAVING COUNT(*) >= 3`. Say which
count you mean: all rows, or distinct orders if a join could duplicate them. Follow-ups:

- *Why not put the status condition in `HAVING`?* It is a row condition. Putting it in `HAVING`
  counts the wrong rows, or fails, because `HAVING` sees groups.
- *How does NULL affect an average?* `AVG` skips NULLs, so `AVG` and `SUM / COUNT(*)` differ.
  Decide which one the question means and say so.

## Connections

- **Prerequisite:** `sql/sql-select-filter-and-sort` (the row pipeline this topic extends).
- **Unlocks:** `sql/sql-joins-and-unions` (where the grain problem appears), `sql/sql-conditional-
  aggregation` (`CASE` inside aggregates), and `sql/sql-top-n-and-latest-row-per-group`.
- **Lineage:** grouped aggregation is the relational counterpart of a fold over a bag. SQL made it
  standard in the 1980s, and the NULL rules above come from that standard.

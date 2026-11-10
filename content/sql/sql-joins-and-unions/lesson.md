---
title: SQL joins and unions
minutes: 12
min_viable: "Read The system and the Worked example, then solve p01."
---

## The system

Three decisions come before any join is written.

1. **Grain of each table.** One row per what? `customers` here has one row per *version* of a customer,
   so a customer can appear twice. `orders` has one row per order.
2. **Grain of the answer.** Write it in one sentence: "one row per current customer."
3. **Which side defines the answer.** Start the `FROM` clause on that side and join the other side onto it.

| Join | Keeps | Use when |
|---|---|---|
| `INNER JOIN` | Rows that match on both sides | Only things present on both sides count |
| `LEFT JOIN` | Every left row; NULLs on the right when nothing matches | The left side defines the answer |
| `RIGHT` / `FULL OUTER JOIN` | Every right row, or every row of both sides | Reconciliation: find what is missing on either side |
| `CROSS JOIN` | Every pair of rows | A deliberate combination, such as a calendar times a product list |
| Anti join: `LEFT JOIN ... WHERE right.key IS NULL`, or `NOT EXISTS` | Left rows with no match | "Customers with no orders" |

Set operations stack result sets with the same number of columns:

| Operator | Result |
|---|---|
| `UNION` | Rows from both queries, duplicates removed (across both and within each) |
| `UNION ALL` | Every row from both queries, duplicates kept |

## Worked example

`customers` (`is_current` marks the live version):

| customer_id | name | is_current |
|---|---|---|
| 1 | Oak Yard | TRUE |
| 1 | Former name | FALSE |
| 2 | Pine Mill | TRUE |
| 3 | Ridge | TRUE |

`orders`: order 10 (customer 1, 50.0), order 11 (customer 1, 5.0), order 12 (customer 2, 5.0),
order 13 (customer 99, 500.0).

1. Grain: one row per current customer. Filter to `is_current` first. Customer 1 now has one row.
2. `LEFT JOIN orders ON orders.customer_id = customers.customer_id`. Customer 1 matches 2 orders,
   customer 2 matches 1, customer 3 matches none and keeps one row with NULL order columns.
   Order 13 has no customer row, so it never appears.
3. `COUNT(o.order_id)` gives 2, 1, 0. `COALESCE(SUM(o.amount), 0.0)` gives 55.0, 5.0, 0.0.

The three mistakes, on the same data:

- **No version filter:** customer 1 joins to 2 customer rows, so 4 orders and 110.0 (both doubled).
- **`COUNT(*)`:** customer 3's NULL-extended row counts as 1 order.
- **Inner join:** customer 3 disappears.

## Why it works

A join returns one row for every pair of rows whose keys match. For one key value, the output
has (left rows with that key) times (right rows with that key) rows. This is the whole mechanism
behind both fan-out and row loss.

A `LEFT JOIN` keeps every left row. An unmatched left row is extended with NULLs on the right. So:

- `COUNT(*)` counts joined rows, including the NULL-extended one, and reports 1 for an empty customer.
- `COUNT(o.order_id)` counts non-NULL keys, which are matched orders only, and reports 0.
- `SUM` over no rows is NULL, not 0, so the total needs `COALESCE`.

See the multiple-tables chapters of Beaulieu, *Learning SQL*, and Molinaro, *SQL Cookbook*.

## Signal vs. noise

- **Signal:** the grain check before every join; choosing the join type from "which side defines
  the answer"; `COUNT` of the right-hand key after a `LEFT JOIN`; where a right-table filter goes
  (`ON`, not `WHERE`); `UNION` vs `UNION ALL`; the anti-join pattern.
- **Noise for now:** `FULL OUTER` syntax variants, `USING`, and `NATURAL JOIN` (avoid it: it joins on
  every shared column name).

## Where it breaks

- **Fan-out.** A one-to-many table joined in before aggregation repeats the rows of the other side.
  Symptom: counts and sums too high. `COUNT(DISTINCT key)` can hide the count problem, but a `SUM`
  stays doubled. Fix: reduce the many side to one row per key first (a filter or a subquery).
- **Right-table filter in `WHERE`.** After a `LEFT JOIN`, `WHERE right.amount > 10` removes the
  NULL-extended rows and makes the join an inner join. Put the condition in `ON` to keep the left rows.
- **NULL keys never match.** `NULL = NULL` is unknown, so orders with a NULL `customer_id` vanish from
  an inner join without an error.
- **`NOT IN` with a NULL.** If the subquery returns any NULL, `x NOT IN (...)` is unknown for every
  `x`, and the query returns no rows. `NOT EXISTS` does not have this problem.
- **Union by position.** `UNION` matches columns by position and takes names from the first branch.
  Swapped columns of the same type raise no error, and the result is wrong.
- **Union cost.** `UNION` must find duplicates. Use `UNION ALL` when the sources cannot overlap.

Event-log categories are dirty keys. In a 1,200-row draw, exact grouping has 31 labels; stripping
spaces and folding case collapses them to 7, losing distinct warehouse codes.

## Interview angle

"Every customer with their order total, including the customers with no orders." Answer shape: state the
grain, start from customers, filter to the current version, `LEFT JOIN` orders, count the order key, and
coalesce the sum. Check first for an inner join that drops customers with no orders. Follow-ups: why
`COUNT(*)` is wrong after a `LEFT JOIN`, and what fan-out does to a sum.

## Connections

- **Requires:** `sql/sql-group-by-and-having` (the grain check is the same idea as "one row per group").
- **Unlocks:** `sql/pandas-merge-and-reshape` (the same join types as `how=`, plus `validate=`),
  and `sql/sql-subqueries-and-ctes` (pre-aggregating a many side in a CTE).

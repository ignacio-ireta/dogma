---
title: SQL ranking window functions
minutes: 12
min_viable: "Read The system and the Worked example, then solve p01."
---

## The system

A window function computes a value for each row from a set of related rows, without collapsing them the way
`GROUP BY` does. The window is set by `OVER`:

`RANK() OVER (PARTITION BY category ORDER BY revenue DESC NULLS LAST)`

- `PARTITION BY` splits the rows into independent groups. Here, one group per category.
- `ORDER BY` sets the order inside a group. Rows with equal `ORDER BY` values are **peers**.

The three ranking functions differ only in how they treat peers:

| Function | Peers share a value? | Next value after a tie | Use when |
|---|---|---|---|
| `ROW_NUMBER` | no: every row gets its own number | no gap, but the tie order is arbitrary unless `ORDER BY` adds a key | exactly one row per group: dedupe, latest row |
| `RANK` | yes | skips: 1, 1, 3 | competition places, where a tie takes two places |
| `DENSE_RANK` | yes | no skip: 1, 1, 2 | "top N distinct values", tiers |

## Worked example

Revenues in one category: 250 (product 101), 250 (103), 180.5 (102), 90 (105), NULL (104). Sorted by revenue descending with NULL last:

| product | revenue | `ROW_NUMBER` | `RANK` | `DENSE_RANK` |
|---|---|---|---|---|
| 101 | 250 | 1 | 1 | 1 |
| 103 | 250 | 2 | 1 | 1 |
| 102 | 180.5 | 3 | 3 | 2 |
| 105 | 90 | 4 | 4 | 3 |
| 104 | NULL | 5 | 5 | 4 |

`ROW_NUMBER` breaks the tie between 101 and 103 by `product_id`, so that the numbers are unique. `RANK` skips to 3 after the tie, and `DENSE_RANK` goes to 2.

Top two by `DENSE_RANK`, including ties: products 101, 103 and 102, since 102 is dense rank 2. Top two by `ROW_NUMBER` would return only 101 and 103.

## Why it works

**Peers are the key.** `RANK` and `DENSE_RANK` treat equal `ORDER BY` values as one place. If you add a tie-breaker to their `ORDER BY`, no two rows are peers, and the tie disappears. Put the tie-breaker only where you want uniqueness, which is `ROW_NUMBER`.

**Evaluation order.** Window functions run after `WHERE`, `GROUP BY` and `HAVING`, but before the final `ORDER BY` and `LIMIT`. A window therefore cannot appear in `WHERE`. Compute it in a subquery or a CTE and filter outside, or use `QUALIFY`, which DuckDB and some other engines support.

**NULL placement.** DuckDB 2.0 sorts NULL last for both `ASC` and `DESC` by default (checked). Other engines differ, so writing `NULLS LAST` makes the answer portable.

## Signal vs. noise

- **Signal:** the `ORDER BY` inside `OVER` decides the answer. Say the tie rule out loud before you write the query.
- **Signal:** state where NULLs sort, and check that each NULL is a peer of the other NULLs.
- **Noise:** frame clauses such as `ROWS BETWEEN` for now. They matter for running totals, which come later.

## Where it breaks

- **Unstable output.** `ROW_NUMBER` with no tie-breaker can return different rows on different runs or engines. Add a unique key.
- **Filtering a window in `WHERE`.** The engine rejects it: "WHERE clause cannot contain window functions". Use a subquery.
- **Text ordering of ids.** Casting an id to text changes the order: `'9'` sorts after `'17'`. Compare numeric ids as numbers.
- **Duplicates after a join.** A join before ranking can duplicate rows, and every rank then counts the copies.

## Interview angle

"Top two products per category." Answer shape: `DENSE_RANK` in a subquery, keep rank at most 2 outside it, and keep ties if the question asks for them. For "latest row per customer", use `ROW_NUMBER` with an explicit tie-breaker, and keep the row numbered 1. State where the NULLs sort. Follow-ups are in `interview.md`.

## Connections

- **Prerequisites:** `sql/sql-group-by-and-having` (aggregates per group), `sql/sql-subqueries-and-ctes` (the derived table that filters a window).
- **Unlocks:** `sql/sql-top-n-and-latest-row-per-group`, `sql/sql-lag-and-lead`, `sql/sql-running-totals-and-frames`.
- **Lineage:** window functions were added to the SQL standard in SQL:2003. The ranking recipes are in Molinaro's SQL Cookbook.
- **References:** [SQL Cookbook, second edition](https://www.oreilly.com/library/view/sql-cookbook-2nd/9781492077435/) locates ranking in §11.9 and windows in appendix A; [DuckDB window documentation](https://duckdb.org/docs/current/sql/functions/window_functions.html) gives the function semantics.

# Customers with three or more completed orders

Write one DuckDB SQL `SELECT` statement, in `submission.sql`, that returns one row for each
customer who has at least three completed orders. Report how many completed orders the customer
has and the total of their completed amounts. Order the customers by total, largest first.

## Contract

- Input: a table `orders` with columns `order_id` (BIGINT), `customer_id` (BIGINT),
  `order_date` (DATE), `status` (VARCHAR) and `amount` (DOUBLE, may be NULL).
- Returns: one row per qualifying customer, with exactly these columns in this order:
  `customer_id`, `completed_orders`, `total_amount`.
- `completed_orders`: the number of orders with `status` equal to `'completed'`, an exact match
  (capitalisation matters). An order whose `amount` is NULL still counts.
- `total_amount`: the sum of `amount` over those same orders. NULL amounts are skipped. If every
  one of a customer's completed orders has a NULL amount, `total_amount` is NULL, not 0.
- A customer qualifies when `completed_orders` is at least 3. Only completed orders count toward
  this threshold. A customer with four orders, two of them completed, does not qualify.
- Order of rows: `total_amount` descending with NULLs last, then `customer_id` ascending.
  Row order is part of the answer.
- If no customer qualifies, return zero rows with the same three columns.
- Must read from `orders`. Do not write the answer as literal rows.

## Examples

On `fixtures/orders.csv`:

```text
customer_id  completed_orders  total_amount
207          5                 440.00
202          3                 300.00
210          3                 300.00
208          3                 180.00
201          4                 145.50
204          3                 100.00
```

Customer 201 has four completed orders. One of them has a NULL amount, and it still counts. Customer
203 has four orders in total but only two are completed, so it is absent. Customers 202 and 210
both total 300.00, so 202 comes first.

A small case: one customer's completed amounts are `(40.0, NULL, 10.0)` and another's are all
NULL:

```text
customer_id  completed_orders  total_amount
1            3                 50.0
2            3                 NULL
```

## Constraints

- Allowed: standard `SELECT`, `FROM`, `WHERE`, `GROUP BY`, `HAVING`, `ORDER BY`, and DuckDB's
  built-in aggregates and operators.
- Not allowed: literal `VALUES` rows or any other hard-coded answer.

## Done looks like

All public tests pass, then **Submit** runs the hidden tests. They build their own `orders`
tables. The hidden tests cover these families: a group of exactly three and of two, non-completed
orders (and other capitalisations of `completed`) that must not count toward the threshold, NULL amounts that count as orders but are
skipped by the sum, a group whose total is NULL, totals that tie, one customer with many orders,
no qualifying customers, and random tables checked against an independent pandas computation.

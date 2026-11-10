# Orders above their customer's average

Write one DuckDB SQL `SELECT` statement, in `submission.sql`, that returns each completed order whose
amount is strictly above the average completed amount of the customer who placed it. Each order is
compared with its own customer's average, not with one average for everyone.

## Contract

- Input: a table `orders` with `order_id` (BIGINT), `customer_id` (BIGINT, may be NULL),
  `order_date` (DATE), `status` (VARCHAR) and `amount` (DOUBLE, may be NULL).
- Amounts are multiples of 0.25, which DOUBLE stores exactly, so sums and averages of equal
  amounts are exact and "strictly greater" has no rounding ambiguity. With arbitrary cents it
  does; the lesson shows why.
- A **candidate** is an order whose `status` is exactly `'completed'`, whose `amount` is not NULL,
  and whose `customer_id` is not NULL. Capitalisation matters.
- `customer_average` of a candidate: the average `amount` over all completed orders of the same
  customer, with NULL amounts skipped (the usual SQL `AVG`). The candidate itself is part of that
  average. Orders with any other status are not.
- Returns: one row for each candidate whose `amount` is strictly greater than its `customer_average`.
  Columns, in this order: `order_id`, `customer_id`, `amount`, `customer_average`.
- An order with a NULL `customer_id` belongs to no customer. It is never returned, and it does not
  take part in any average.
- Order of rows: `order_id` ascending.
- If no row qualifies, return zero rows with the same four columns.
- Must read from `orders`. Do not write the answer as literal rows.

## Examples

On `fixtures/orders.csv`:

```text
order_id  customer_id  amount  customer_average
3         501          80.0    50.0
7         503          30.0    20.0
10        504          120.0   110.0
13        505          15.0    11.67
```

The last average is 35/3, shown rounded. Customer 501 has completed amounts 20, 50 and 80, so its
average is 50 and only order 3 qualifies. Its refunded order of 300 is not counted. Customer 503 has
amounts 10, NULL and 30. The NULL is skipped, so its average is 20. Customer 502 has one completed
order, which equals its own average, so it is not returned.

A small case, with one customer whose completed amounts are 0.0, 0.0, 0.0 and 90.0:

```text
order_id  customer_id  amount  customer_average
4         8            90.0    22.5
```

The average is 90 / 4 = 22.5. The three zero orders are not above it.

## Constraints

- Allowed: `SELECT`, `FROM`, `WHERE`, `GROUP BY`, `HAVING`, `ORDER BY`, joins, subqueries, common
  table expressions (`WITH`), and DuckDB's built-in aggregates and window functions.
- Not allowed: literal `VALUES` rows or any other hard-coded answer.

## Done looks like

All public tests pass, then **Submit** runs the hidden tests. They build their own `orders` table.
The families they cover: an order equal to its own customer's average, an average that must ignore
pending, refunded and cancelled orders and capitalised variants of `completed`, NULL amounts that are
skipped and never returned, a customer with a single known amount, orders with no customer, a
customer with several qualifying orders (one row per order, not per customer), output ordered by
order id even when rows were inserted out of order, a result with no row, and random tables checked
against an independent pandas computation.

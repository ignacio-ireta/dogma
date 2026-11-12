# Customer order totals, including customers with no orders

Write one DuckDB SQL `SELECT` statement, in `submission.sql`, that returns one row for every
current customer: how many orders they placed and the total of those orders. A customer with no
orders still gets a row, with `0` and `0.0`. An order whose customer is not in the customer table
must not change any result.

## Contract

- Input: two tables.
  - `customers`: `customer_id` (BIGINT), `customer_name` (VARCHAR), `is_current` (BOOLEAN). The
    table keeps one row per version of a customer. Each `customer_id` has exactly one row with
    `is_current = TRUE`. Older versions have `FALSE`.
  - `orders`: `order_id` (BIGINT), `customer_id` (BIGINT), `order_date` (DATE), `amount` (DOUBLE,
    never NULL).
- Returns: one row per current customer, with exactly these columns in this order:
  `customer_id`, `customer_name`, `order_count`, `total_amount`.
  - `customer_name`: the name on the current version.
  - `order_count`: the number of orders with that `customer_id`. An integer, `0` when there are none.
  - `total_amount`: the sum of those orders' `amount`. A float, `0.0` when there are none. Never NULL.
    Sum in DOUBLE, the type of `amount`: a narrower type such as REAL keeps only about seven
    significant digits, so cents drift in large totals.
- Orders whose `customer_id` has no current row in `customers` are excluded from every count and
  every sum.
- Row order: `total_amount` descending, then `customer_id` ascending. Row order is part of the answer.
- If `customers` is empty, return zero rows with the same four columns.
- Must read from `customers` and `orders`. Do not write the answer as literal rows.

## Examples

Small case, with `customers` on the left and `orders` on the right:

```text
customer_id  customer_name  is_current        order_id  customer_id  amount
7            Oak Yard       TRUE              1         7            12.5
8            Pine Mill      TRUE              2         7            7.5
                                              3         99           40.0   (no customer 99)

customer_id  customer_name  order_count  total_amount
7            Oak Yard       2            20.0
8            Pine Mill      0            0.0
```

On `fixtures/customers.csv` and `fixtures/orders.csv`, the first and last rows of the answer are:

```text
customer_id  customer_name            order_count  total_amount
104          Maple Studio             2            340.0
...
108          Copper Tailors           0            0.0
110          Lantern Garage           0            0.0
112          Quarry Print Shop        0            0.0
```

Customer 101 has two versions. Only the current one is returned, once, with its three orders.
Customers 105 and 109 both total 250.0, so 105 comes first. The orders for customers 113 and 120
are not in the customer table, so they do not appear.

Large totals keep their cents. A customer with orders of 1234.56 and 789.01 totals 2023.57. A
query that casts `amount` to REAL before summing returns 2023.570068, which is off past the fourth
decimal and outside the tolerance.

## Constraints

- Allowed: `SELECT`, `FROM`, `JOIN`, `WHERE`, `GROUP BY`, `ORDER BY`, `COALESCE`, and DuckDB's
  built-in aggregates.
- Not allowed: literal `VALUES` rows or any other hard-coded answer.

## Done looks like

All public tests pass, then **Submit** runs the hidden tests. They build their own `customers` and
`orders` tables, so a query tuned to the fixture fails. The hidden tests cover these families:
customers with no orders (count and total), customers with several versions (one row each, with the
current name, and no order counted twice), orders for customers that do not exist, equal and zero
totals with the tie-break, an empty orders table, no customers at all, and random tables checked
against an independent pandas computation, and cent amounts summed to large totals, held to the same
tolerance (no narrower type).

The hidden tests draw their bulk inputs from the real dataset `online_retail` (see
`docs/DATASETS.md`).

# Customers above the average completed spend

Write one DuckDB SQL `SELECT` statement, in `submission.sql`, that returns each customer whose total
completed spend is strictly above the average total completed spend of the buyers. Report the
customer's region, their total and the average. Order the rows by total, largest first.

## Contract

- Input: a table `customers` with `customer_id` (BIGINT) and `region` (VARCHAR), one row per
  customer. A table `orders` with `order_id` (BIGINT), `customer_id` (BIGINT, may be NULL),
  `order_date` (DATE), `status` (VARCHAR) and `amount` (DOUBLE, may be NULL).
- Amounts are multiples of 0.25, which DOUBLE stores exactly, so sums and averages of equal
  amounts are exact and "strictly greater" has no rounding ambiguity. With arbitrary cents it
  does; the lesson shows why.
- A **buyer** is a customer with at least one order whose `status` is exactly `'completed'`.
  Capitalisation matters.
- `total_spend` of a buyer: the sum of `amount` over the buyer's completed orders. NULL amounts are
  skipped. If every completed amount of the buyer is NULL, `total_spend` is NULL.
- `average_spend`: the average of `total_spend` over the buyers, with the usual SQL `AVG` rule: NULL
  totals are skipped. A customer with no completed order is not a buyer, so it is not in the
  average and it does not count as 0. The value is the same on every returned row. If no buyer has
  a known total, it is NULL.
- Returns: one row for each buyer whose `total_spend` is strictly greater than `average_spend`.
  Columns, in this order: `customer_id`, `region`, `total_spend`, `average_spend`.
- A buyer whose total equals the average is not returned.
- An order with a NULL `customer_id`, or whose `customer_id` has no row in `customers`, belongs to no
  customer. It is not returned and it is not counted anywhere, including in `average_spend`.
- Order of rows: `total_spend` descending, then `customer_id` ascending.
- If no row qualifies, return zero rows with the same four columns.
- Must read from `customers` and `orders`. Do not write the answer as literal rows.

## Examples

On `fixtures/customers.csv` and `fixtures/orders.csv`:

```text
customer_id  region  total_spend  average_spend
305          north   300.00       180.00
306          south   300.00       180.00
303          north   240.00       180.00
```

The buyers with a known total add up to 1440.00 across eight customers, so the average is 180.00.
Customer 307 totals exactly 180.00, so it is not returned. Customer 308 has only NULL completed
amounts: it is a buyer with a NULL total, so it is left out of the average. Customers 305 and 306
tie at 300.00, so 305 comes first. Customer 310 has only a pending order and customer 311 has no
orders. Neither is a buyer.

A small case, with `customers` = (1, north), (2, south), (3, north) and these completed orders:
customer 1 has amounts 40.0, NULL and 10.0; customer 2 has 20.0; customer 3 has none.

```text
customer_id  region  total_spend  average_spend
1            north   50.0         35.0
```

The buyers' totals are 50.0 and 20.0, so the average is 35.0. Customer 3 is not a buyer.

## Constraints

- Allowed: `SELECT`, `FROM`, `WHERE`, `GROUP BY`, `HAVING`, `ORDER BY`, joins, subqueries, common
  table expressions (`WITH`), and DuckDB's built-in aggregates and window functions.
- Not allowed: literal `VALUES` rows or any other hard-coded answer.

## Done looks like

All public tests pass, then **Submit** runs the hidden tests. They build their own `customers` and
`orders` tables. The families they cover: a customer exactly at the average, a buyer whose completed
amounts are all NULL, a buyer with some NULL amounts, customers with no completed order (pending
only, cancelled only, or no order at all), non-completed statuses and a capitalised variant that must
not count, orders with no customer and orders whose customer has no row, ties in total broken by customer id with large non-sequential
ids, an average over customer totals rather than over order amounts, a result with no row (zero rows
with the four columns), and random tables checked against an independent pandas computation.

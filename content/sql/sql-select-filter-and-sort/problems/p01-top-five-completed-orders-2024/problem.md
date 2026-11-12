# Top five completed orders of 2024

Write one DuckDB SQL `SELECT` statement, in `submission.sql`, that returns the five largest
completed orders placed in 2024. An order qualifies only if its status is exactly `completed`,
its date is in 2024, and its amount is known. Rank qualifying orders by amount, and keep the top
five.

## Contract

- Input: a table `orders` with columns `order_id` (BIGINT), `customer_id` (BIGINT),
  `order_date` (DATE), `status` (VARCHAR) and `amount` (DOUBLE, may be NULL).
- Returns: one row per kept order, with exactly these columns in this order: `order_id`,
  `customer_id`, `order_date`, `amount`.
- Qualifies: `status` equals `'completed'` exactly (spelling and capitalisation matter);
  `order_date` is in 2024, so both 2024-01-01 and 2024-12-31 count; `amount` is not NULL. A zero
  amount is a known amount and qualifies.
- Order of rows: `amount` descending, then `order_id` ascending. Row order is part of the answer.
- Keeps at most five rows. If fewer than five orders qualify, return all of them. If none
  qualify, return zero rows with the same four columns.
- Must read from `orders`. Do not write the answer as literal rows.

## Examples

On `fixtures/orders.csv`, loaded as the `orders` table with `read_csv_auto`:

```text
order_id  customer_id  order_date  amount
20        114          2024-12-31  330.00
12        109          2024-06-18  310.40
14        105          2024-07-25  310.40
17        111          2024-01-01  275.00
7         106          2024-03-21  250.00
```

Ties on amount are broken by `order_id`. Suppose these completed 2024 orders exist:

```text
order_id  amount
9         300.00
4         300.00
2         80.00
```

The answer is order 4, then order 9, then order 2. Order 4 comes first because it has the lower
`order_id` among the 300.00 orders.

## Constraints

- Allowed: standard `SELECT`, `FROM`, `WHERE`, `ORDER BY`, `LIMIT`, and DuckDB's built-in
  operators and date literals.
- Not allowed: literal `VALUES` rows or any other hard-coded answer.

## Done looks like

All public tests pass, then **Submit** runs the hidden tests. They build their own `orders`
tables, so a query tuned to the fixture fails. The hidden tests cover these families:
year boundaries (the first and last day of 2024 in, the days either side out), status values
other than `completed` (including the same word with another capitalisation), NULL amounts, a
zero amount, ties at the cutoff, fewer than five matches, no matches, and random tables checked
against an independent pandas computation.

The hidden tests draw their bulk inputs from the real dataset `online_retail` (see
`docs/DATASETS.md`).

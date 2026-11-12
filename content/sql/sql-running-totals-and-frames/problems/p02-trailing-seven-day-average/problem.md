# Trailing seven-day average of daily sales

A bakery chain has three shops. A shop has no row for a day it was closed, so the dates in the
table have gaps. For each shop and each day that has a row, return the average daily revenue over
the seven calendar days ending on that day.

**Frame: calendar, not row count.** The window is a RANGE frame on `sales_date` with six days
preceding and the current row, so it is measured in dates. A ROWS frame of six preceding rows is
wrong here: on a shop with closed days it reaches back more than seven calendar days.

**Missing days: absent.** A closed day is not zero revenue. It has no row, so it is neither added
as zero nor counted in `days_in_window`.

**Output at a glance:**

| Column | DuckDB type | Meaning |
|---|---|---|
| `store_id` | BIGINT | Shop |
| `sales_date` | DATE | The row's own date |
| `trailing_average_cents` | DOUBLE | Mean `revenue_cents` over the window |
| `days_in_window` | BIGINT | Rows in the window; at least 1 |

## Contract

Table `daily_sales` has columns:

| Column | DuckDB type | Rules |
|---|---|---|
| store_id | BIGINT | Non-null |
| sales_date | DATE | Non-null; unique within a store |
| revenue_cents | BIGINT | Non-null; zero or positive |

For each row `r` with store `s` and date `d`, the window is every row of store `s` whose
`sales_date` `x` satisfies `d - 6 <= x <= d` (seven calendar days, both ends included). The
window always contains row `r` itself.

Return one row per input row, with the output columns and types in the table above, in this order:
`store_id`, `sales_date`, `trailing_average_cents`, `days_in_window`.

- Final output order is `store_id ASC, sales_date ASC`.
- Do not zero-fill closed days, and do not count them.
- An empty table returns no rows, with the stated column names and types.
- Do not change the input table.

Averages are compared with an absolute tolerance of 1e-9 and a relative tolerance of 1e-12.

## Examples

Store 1 has rows on 1 January (100), 3 January (300), 7 January (7), 9 January (50) and
10 January (70). Rows for 2, 4, 5, 6 and 8 January do not exist.

```text
store_id sales_date  revenue_cents
1        2026-01-01  100
1        2026-01-03  300
1        2026-01-07  7
1        2026-01-09  50
1        2026-01-10  70
=> (1, 2026-01-01, 100.0,            1)
   (1, 2026-01-03, 200.0,            2)
   (1, 2026-01-07, 135.666666...,    3)   -- 1 Jan to 7 Jan: 100, 300, 7
   (1, 2026-01-09, 119.0,            3)   -- 3 Jan to 9 Jan: 300, 7, 50
   (1, 2026-01-10, 42.333333...,     3)   -- 4 Jan to 10 Jan: 7, 50, 70
```

A ROWS frame of six preceding rows would give 10 January the last five rows (105.4 over a span of
ten days), which is why the window is defined by calendar dates.

```text
store_id sales_date  revenue_cents
2        2026-02-28  999
=> (2, 2026-02-28, 999.0, 1)
```

## Constraints

- Allowed: a read-only DuckDB SQL query. Window functions, joins, CTEs and subqueries are allowed.
- Not allowed: changing the `daily_sales` table. Do not add rows for closed days.
- The grader checks results, not syntax.

## Done looks like

The result matches an independent oracle on seeded shops with random gaps across several months,
and on a fixture. Hidden cases cover a window that ends exactly six days after an earlier row
(included) and seven days after (excluded), gaps longer than the window, stores with the same
dates that must not mix, a store with one day, dates across a month and a year boundary, large
revenue values, exact output types and final ordering, and an empty table.

The hidden tests draw their bulk inputs from the real dataset `online_retail` (see
`docs/DATASETS.md`).

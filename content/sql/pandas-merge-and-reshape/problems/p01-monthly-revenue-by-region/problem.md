# Monthly revenue by region

Implement `monthly_revenue_by_region(orders, customers)`. It returns a wide table of revenue per
region (rows) and per month (columns). Each order is attributed to its customer's region. Orders from
customers that are not in the table go to the region `"unknown"`. Orders with no amount are dropped,
not filled with zero.

## Contract

- `orders`: a `pandas.DataFrame` with columns `order_id`, `customer_id`, `order_date`, `amount`.
  `order_date` is a `datetime64` column or ISO-8601 date strings such as `"2024-03-05"`. `amount` is
  numeric, integers or floats, and may contain NaN.
- `customers`: a `pandas.DataFrame` with columns `customer_id` and `region` (strings).
  `customer_id` is unique.
- Returns: a `pandas.DataFrame` with `float64` values in every cell.
  - Index: region labels (strings), sorted ascending, named `"region"`.
  - Columns: month labels `"YYYY-MM"` (strings), sorted ascending, named `"month"`.
  - Each cell is the sum of the known amounts for that region and month. A region-month pair with
    no known amount has the cell `0.0`, never NaN.
- The rules, in order:
  1. Left-merge `orders` onto `customers` by `customer_id` with `validate="many_to_one"`. A duplicate
     `customer_id` in `customers` raises `ValueError`. pandas raises `MergeError`, which is a
     `ValueError` subclass; let it propagate, do not catch it.
  2. An order whose customer is not in `customers` gets the region `"unknown"`. A customer who is in
     `customers` but whose `region` is missing (NaN) also gets `"unknown"`, and its orders are kept.
     `"unknown"` is an ordinary label, so it sorts like any other region name.
  3. An order with NaN `amount` is dropped before grouping. Its amount is unknown, not zero, so it
     creates no region row and no month column of its own.
  4. The month of an order is its calendar month, written `"YYYY-MM"`.
- If no order has a known amount, return an empty frame with no rows and no columns. Its index is
  named `"region"` and its columns are named `"month"`.
- Must not mutate `orders` or `customers`.
- Raises: `ValueError` when `customers` has a duplicate `customer_id`.

## Examples

With `customers` mapping customer 10 to `North` and customer 20 to `South`, and `orders`:

```text
order_id  customer_id  order_date  amount
1         10           2024-01-05  100.0
2         10           2024-02-10  50.0
3         20           2024-01-20  30.0
4         99           2024-01-22  40.0      (customer 99 is not in customers)
5         20           2024-02-02  NaN       (unknown amount: dropped)
6         10           2024-02-14  25.0

region   2024-01  2024-02
North    100.0    75.0
South    30.0     0.0
unknown  40.0     0.0
```

Customer 10 has two February orders, 50.0 and 25.0, which sum to 75.0. South's February order has no
amount, so South's February cell is `0.0`. Order 4 is not in `customers`, so it lands in `unknown`.

A second case: region `East` (customer 30) has one order, with a NaN amount. `East` has no row at all.

```text
order_id  customer_id  order_date  amount
1         10           2024-01-05  100.0
2         30           2024-03-01  NaN

region  2024-01
North   100.0
```

An unknown amount does not create a region with a zero total, and it does not create the 2024-03
column.

The interview-prep orders and customers datasets provide order amounts, dates, and city labels, and
about one customer in twenty has no city. Use each customer's city as this problem's region label.
A customer with no city is in `customers`, so its orders go to `"unknown"`, as the second rule says.

## Constraints

- Allowed: `pandas` and `numpy`, and the standard library.
- Not allowed: writing the result as a literal table, or any other hard-coded answer.

## Done looks like

All public tests pass, then **Submit** runs the hidden tests. They build their own `orders` and
`customers` frames. The hidden tests cover these families: orders from unknown customers, duplicate
customer ids (must raise, not double the revenue), NaN amounts (dropped, not filled), regions with
only unknown amounts, missing cells that must be `0.0`, integer amounts giving float cells, `datetime64`
and string dates giving the same months, months sorted across a year boundary, an empty result with
its names, inputs left unchanged, customers with a missing region (interview-prep, NaN city), lognormal
amounts with missing values, and interview-prep rows checked against a pandas merge and groupby oracle.

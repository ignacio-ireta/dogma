# Customer order features

Implement `customer_order_features(orders)`. It returns one row per order, with four features that
place each order in its customer's history: the order's sequence number, its share of the customer's
total spend, the days since the customer's previous order, and the customer's running total. Each
feature is a per-customer computation, so the work is a groupby on `customer_id`, done after the rows
are sorted in time.

## Contract

- `orders`: a `pandas.DataFrame` with columns `order_id` (int64, unique), `customer_id` (int64),
  `order_date` (datetime64, or ISO-8601 date strings such as `"2024-03-05"`, dates only with no time
  part) and `amount` (finite float64, signed amounts including refunds and zero allowed). It may have zero rows.
- Returns: a new `pandas.DataFrame` with one row per input row, sorted by `customer_id`, then
  `order_date`, then `order_id`, all ascending. Its index is a fresh 0 to n-1 RangeIndex. Its columns,
  in this order, are:
  - `order_id`: int64, as input.
  - `customer_id`: int64, as input.
  - `order_date`: datetime64, the input dates parsed.
  - `amount`: float64, as input.
  - `order_seq`: int64. 1 for the customer's first order in sort order, 2 for the next, and so on.
  - `share_of_customer_total`: float64. `amount` divided by the sum of that customer's amounts over all
    of their orders. `NaN` when that sum is `0.0`.
  - `days_since_previous_order`: float64. The calendar days from the customer's previous order date to
    this order's date. `0.0` when two orders share a date. `NaN` for the customer's first order.
  - `customer_running_total`: float64. The sum of that customer's amounts up to and including this
    order, in sort order.
- The dtypes above hold for every input, including an empty one.
- Must not mutate `orders`: no added or removed columns, no reordered rows, no changed dtypes.
- Raises: nothing. Bad input is the caller's problem; the function does not catch errors.

## Examples

Input, in arrival order. Customer 10 has two orders on 2024-01-15, and the lower `order_id` comes
first in the sort:

```text
order_id  customer_id  order_date  amount
4         10           2024-02-01  10.0
7         10           2024-01-15  40.0
2         10           2024-01-15  20.0
9         20           2024-01-03  30.0
5         20           2024-03-01  0.0
```

Output, with shares shown to 4 places (the function returns full precision):

```text
order_id  customer_id  order_date  amount  order_seq  share_of_customer_total  days_since_previous_order  customer_running_total
2         10           2024-01-15  20.0    1          0.2857                   NaN                        20.0
7         10           2024-01-15  40.0    2          0.5714                   0.0                        60.0
4         10           2024-02-01  10.0    3          0.1429                   17.0                       70.0
9         20           2024-01-03  30.0    1          1.0                      NaN                        30.0
5         20           2024-03-01  0.0     2          0.0                      58.0                       30.0
```

Customer 10's total is 70.0, so its shares are 20/70, 40/70 and 10/70.

A customer whose orders all total zero has NaN shares, not 0 and not an error:

```text
order_id  customer_id  order_date  amount
11        30           2024-05-01  0.0
12        40           2024-05-02  0.0
```

```text
order_id  customer_id  order_date  amount  order_seq  share_of_customer_total  days_since_previous_order  customer_running_total
11        30           2024-05-01  0.0     1          NaN                      NaN                        0.0
12        40           2024-05-02  0.0     1          NaN                      NaN                        0.0
```

## Constraints

- Allowed: `pandas` and `numpy`, and the standard library.
- Not allowed: `duckdb`, or any other SQL engine, and any hard-coded answer or literal result rows.

## Done looks like

All public tests pass, then **Submit** runs the hidden tests. They build their own `orders` frames,
mostly shuffled, and compare the result with an independent DuckDB window-function computation. The
hidden tests cover these families: rows that arrive out of time order, several customers interleaved,
two orders on the same date (the order id breaks the tie), a gap of zero days, a first order with no
gap, a customer whose total is zero (including refunds cancelling spend), string dates and datetime values giving the same result, an empty
input keeping the same columns and dtypes, the dtype of every column, a fresh index, inputs left
unchanged, and random tables.

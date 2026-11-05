---
title: pandas groupby, aggregate and transform
minutes: 12
min_viable: "Read The system and the Worked example, then solve p01."
---

## The system

`groupby` is split-apply-combine. It splits rows into groups by a key, applies a function to each
group, and combines the results. The three methods differ by the shape of what they return.

| Method | Returns | Index of the result | Use it for |
|---|---|---|---|
| `agg` | one row per group | the group keys | a summary per customer |
| `transform` | one value per input row | the input row index | a per-row value that uses a group statistic |
| `apply` | whatever your function returns | depends on `group_keys` | anything the first two cannot express |

`cumcount`, `cumsum` and `shift` return one value per row, like `transform`. Two rules come before
any per-group step. The rows must be in the order you mean (inside a group, the order of the rows is
the order of the calculation). And you must know which shape each feature needs.

## Worked example

Customer 10 has three orders: order 2 (2024-01-15, 20.0), order 7 (2024-01-15, 40.0) and order 4
(2024-02-01, 10.0). Four features for these rows.

1. **Sort.** By customer, then date, then order id: 2, 7, 4. Order 2 comes first because it shares a
   date with 7 and has the lower id.
2. **Sequence.** `groupby("customer_id").cumcount() + 1` gives 1, 2, 3.
3. **Share.** `transform("sum")` gives 70.0 on every row of the customer. Dividing gives
   20/70 = 0.2857, 40/70 = 0.5714 and 10/70 = 0.1429.
4. **Gap.** `groupby("customer_id")["order_date"].shift()` gives NaT, 01-15 and 01-15. Subtracting
   and taking `.dt.days` gives NaN, 0 and 17.
5. **Running total.** `groupby("customer_id")["amount"].cumsum()` gives 20, 60 and 70.

Each column has one row per order, so each one is assigned straight back to the frame. No merge is needed.

## Why it works

`transform` returns a Series with the same index as the input, so it lines up with the rows it came
from. `agg` returns one row per key, so writing it back needs a merge on the key. Both are correct.
The transform route avoids the join and keeps the row count fixed (McKinney, *Python for Data
Analysis*, the chapter on data aggregation and group operations).

Inside a group, `cumcount` counts 0, 1, 2 in the row order of that group, and `cumsum` and `shift`
also follow row order. So the sort is what makes the sequence and the gaps mean "time order". The
`sort=True` default orders the groups themselves, not the rows inside them.

## Signal vs. noise

- **Signal:** the three shapes, and which one each feature needs. Sort before the per-group steps.
  The dtype of every output column.
- **Signal:** the defaults that change results. `dropna=True` drops rows whose group key is NaN, and
  `sort=True` orders the groups.
- **Noise for now:** multi-level column names from `agg` with a list of functions. Named aggregation
  avoids them.

## Where it breaks

- **Unsorted rows.** A running total or a gap computed before the sort is correct only for the input
  order, which is usually the arrival order. Sort first.
- **Dropped NaN keys.** With the default `dropna=True`, a row whose `customer_id` is NaN gets NaN from
  `transform` and `cumcount`. Pass `dropna=False` if those rows matter.
- **Dtypes that depend on the data.** `rank(method="first")` returns float64, even when the values are
  whole numbers. `.dt.days` is float64 when a NaT is present but int64 for an empty frame. Cast each
  output explicitly, so the dtype does not change with the input.
- **Zero totals.** Refunds can cancel positive spend, so a zero customer total need not mean each
  amount is zero. Mask zero totals before dividing to enforce NaN shares; otherwise nonzero amounts
  divided by zero produce infinities.
- **`apply` as the default.** `apply` calls your function once per group in Python; built-in grouped
  reductions avoid that callback overhead. Check the result's index and empty-input shape before
  assigning or joining it. Use `apply` when no built-in method fits.

## Interview angle

"For each order, add the customer's order number, their share of spend, and the days since their
previous order." Answer shape: sort by customer, date and order id. Then `groupby`, with `cumcount() + 1`
for the number, `transform("sum")` for the share, and a grouped `shift` for the gap. Name the first-order
NaN and the zero-total NaN before the interviewer asks. Follow-ups:

- *Why not `agg` and merge?* It is also correct. `transform` keeps the index, so there is no join to
  check, and no key can be duplicated.
- *When would you use `apply`?* When the per-group result is not one value per row and no built-in
  method fits. It costs one Python call per group.

Named aggregation makes a summary's output columns explicit:
`orders.groupby("customer_id").agg(total_spend=("amount", "sum"), order_count=("order_id", "size"))`.
It has one row per customer, so it needs a merge to attach those columns to orders.

## Connections

- **Requires:** `sql/pandas-merge-and-reshape`. The merge checks the key cardinality, and this topic
  computes per-key features on the merged result.
- **Unlocks:** time-series features, where the same shift and cumsum pattern runs over a date index.
- **Same idea in SQL:** `ROW_NUMBER`, `SUM(...) OVER (PARTITION BY ...)` and `LAG` compute these
  features in SQL. The problem's hidden tests use them as an independent oracle.

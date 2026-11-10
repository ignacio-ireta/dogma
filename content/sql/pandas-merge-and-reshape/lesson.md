---
title: pandas merge and reshape
minutes: 12
min_viable: "Read The system and the Worked example, then solve p01."
---

## The system

Three decisions, in this order, cover most merge and reshape work.

1. **Join type** (`how=`): which rows survive.

| `how=` | Keeps | Use when |
|---|---|---|
| `"inner"` | Keys present on both sides | Only matched rows answer the question |
| `"left"` | Every left row; right columns NaN when unmatched | The left frame defines the answer |
| `"outer"` | Every row of both sides | Reconciling two sources |

2. **Key cardinality** (`validate=`): `"many_to_one"` means the right key is unique. Use it whenever the
   right side is a lookup table. A violation raises `MergeError`, a `ValueError` subclass.
3. **Missing values**: fill or drop, decided per column and stated. Fill when NaN means "unknown
   category" (a label). Drop when NaN means "unknown quantity" (an amount).

Reshaping is one of three operations:

| Goal | Tool | Notes |
|---|---|---|
| Long to wide, unique pairs | `pivot` | Raises on duplicate (index, column) pairs |
| Long to wide, aggregate | `pivot_table(aggfunc=..., fill_value=...)` | Default aggregation is `mean` |
| Long to wide, explicit | `groupby(...).sum().unstack(..., fill_value=...)` | Shows each step |
| Wide to long | `melt(id_vars=..., var_name=..., value_name=...)` | Turns columns back into rows |

## Worked example

`customers` maps customer 10 to `North` and customer 20 to `South`. `orders` has six rows, including
customer 99 (not in `customers`) and one amount that is NaN.

1. **Merge.** `orders.merge(customers, how="left", validate="many_to_one")` keeps all six rows. Order 4
   (customer 99) gets NaN for `region`, and `fillna("unknown")` gives it a label.
2. **Drop unknown amounts.** Order 5 has NaN for `amount`. It is dropped, not filled, so it adds nothing
   to South's February cell, which stays 0.0.
3. **Group and reshape.** Sum the known amounts per (region, month), then `unstack("month",
   fill_value=0.0)`:

| region | 2024-01 | 2024-02 |
|---|---|---|
| North | 100.0 | 75.0 |
| South | 30.0 | 0.0 |
| unknown | 40.0 | 0.0 |

Each cell has a visible reason. North's February cell is 50.0 + 25.0. South's February cell is 0.0
because its only February order has no amount.

## Why it works

A merge returns one row for each pair of rows whose keys match. If the right key is unique, each left row
matches at most one right row, so the row count stays the same. `validate="many_to_one"` turns that
assumption into a check, so a violation fails loudly. A useful sanity check after any merge is to compare
row counts: `len(merged) == len(left)` for a many-to-one merge.

`groupby` splits rows into groups and aggregates each one. `unstack` moves one index level into columns.
Together they are a pivot with an explicit aggregation step, and `fill_value` writes the value for pairs
that have no rows. Sorting the index and columns after the reshape makes the output order part of the
contract instead of an accident of the input order.

The value type matters too. Summing integers gives integers, and filling with a float in an integer
column raises a warning in recent pandas. Cast to `float64` before reshaping.

## Signal vs. noise

- **Signal:** the join type, `validate=`, the fill-or-drop decision for each column, the names of the
  index and columns, and an explicit aggregation (`sum` or `mean`).
- **Signal:** the row count before and after each merge.
- **Noise for now:** `merge_asof`, `combine_first` and multi-index column juggling. Learn them when a
  problem needs them.

## Where it breaks

- **Duplicate keys.** Without `validate=`, a duplicate lookup key repeats every order. Revenue doubles
  and no error appears.
- **NaN keys match NaN keys.** In pandas, two NaN join keys match each other. SQL does not do this.
  Check for missing keys before the merge.
- **`groupby` drops NaN keys.** The default `dropna=True` removes the group with a NaN key, and
  nothing warns. Fill the key with a label first if those rows matter.
- **Fill before drop.** `fillna(0)` on an amount before grouping creates rows and columns for months
  that had only unknown amounts. The fix is to drop, then reshape.
- **`pivot_table` defaults.** `aggfunc` defaults to `mean`, and all-NaN columns are dropped. A monthly
  total with the default can silently report averages, and a missing month can disappear.
- **`pivot` duplicates.** `pivot` raises `ValueError` on duplicate (index, column) pairs. Aggregate
  first, or use `pivot_table`.

The interview-prep customers CSV has 50 repeated customer ids. Customer C02413 appears twice, and
its 11 known-amount orders total 598.36; an unchecked merge on that key doubles it to 1,196.72.
Keep the many-to-one check so the lookup-key defect fails loudly.

## Interview angle

"Revenue by region and month, with orders that have no matching customer." Answer shape: left merge with
`validate="many_to_one"`, an explicit label for unmatched regions, drop unknown amounts, then a `groupby`
and `unstack` with `fill_value=0.0`. Name the failure you check first: a duplicate lookup key that
doubles revenue. Follow-ups: why an inner merge loses money silently, and how `pivot_table` differs from
`groupby` plus `unstack`.

## Connections

- **Requires:** `sql/sql-joins-and-unions`. The join types and the grain check are the same ideas in SQL.
- **Unlocks:** `sql/pandas-groupby-aggregate-and-transform` (the groupby half of this lesson, in depth),
  and time-series work, where reshaping to a date index comes first.
- **Reference:** McKinney, *Python for Data Analysis*, the chapter on data wrangling: join, combine and
  reshape, covers merge types and reshaping with these functions.

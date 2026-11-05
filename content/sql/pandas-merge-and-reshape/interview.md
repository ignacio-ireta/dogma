# Interview notes: pandas merge and reshape

## Q1. Revenue by region and month. Some orders have customers missing from the lookup table. How do you handle it?

**60-second answer.** I left-merge orders onto customers, so no order is lost. I pass `validate="many_to_one"`,
so a duplicate customer id raises instead of doubling revenue. The region of an unmatched order is
missing, and I fill that with an explicit label such as "unknown", so the revenue is kept and visible.
Then I group by region and month, sum the amounts, and unstack the month into columns with a fill of 0.0,
so every cell is a number. The one decision I state out loud is what to do with a missing amount. I drop
it, because an unknown amount is not a zero amount, and a dropped order should not create an empty row.

**Follow-ups.**
- *Why not an inner merge?* It silently removes the orders with no customer, and the totals look smaller
  than they are.
- *How would you check the result?* Compare the row total before and after the merge, and compare the
  grand total against a plain sum of the known amounts.

## Q2. The totals doubled after a merge. Why?

**60-second answer.** A merge repeats the left rows once for every match on the right. If the right
key is not unique, each order appears once per matching row. I check the key's cardinality before
the merge, by counting duplicates, or I let `validate=` do it. The fix is to make the lookup side unique
first, by choosing the current record or aggregating it, and then merge.

**Follow-ups.**
- *What does an indicator column tell you?* `indicator=True` adds `_merge` with values `both`,
  `left_only` and `right_only`. It is the quickest way to see unmatched rows.
- *Does pandas match NaN keys?* Yes, NaN matches NaN in a merge. I check for missing keys first.

## Q3. How do you reshape a long table into a wide one, and when do you use melt?

**60-second answer.** `pivot` reshapes unique pairs and raises on duplicates. `pivot_table` aggregates,
and its default is `mean`, so I pass `aggfunc="sum"` for totals and a `fill_value` for the gaps. The
`groupby` followed by `unstack` is the most explicit form, and I use it when the aggregation needs
several steps. `melt` goes the other way: it turns month columns back into rows, one row per region and
month. I name the index and columns, because a wide table without names is hard to join again.

**Follow-ups.**
- *Why does `pivot_table` drop columns?* It drops all-NaN columns and rows by default, so an empty
  month can disappear.
- *What dtype do the gaps have?* A fill of 0.0 gives floats. On an integer column, pandas 3 warns
  when a float fill is applied, so cast to float64 before the reshape.

## Common mistakes
- An inner merge that quietly drops orders with no matching customer.
- No `validate=`, so a duplicate lookup key doubles revenue without an error.
- Filling a missing amount with 0 before grouping, which creates rows and columns for months that had no
  known revenue.
- `pivot_table` with its default `mean`, reported as a total.
- A `groupby` that drops NaN keys silently, so the unknown group vanishes.

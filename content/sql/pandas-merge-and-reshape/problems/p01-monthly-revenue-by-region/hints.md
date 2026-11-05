## Hint 1 — concept

Think of the table as two questions: which region does each order belong to, and which month is it in?
Answer both as new columns first, then group, then reshape. Keep the three decisions separate: an order
with no customer, an order with no amount, and an empty month slot each need their own rule.

## Hint 2 — formula / approach

Left-merge orders onto customers with `validate="many_to_one"`, then fill the missing region with
`"unknown"`. Drop rows whose `amount` is NaN before grouping, and do not fill them. Group by region and
the `"YYYY-MM"` month, sum the amounts, and unstack the month into columns with `fill_value=0.0`.

## Hint 3 — code shape

```python
attributed = orders.merge(..., how="left", validate=...)  # step 1: region per order
attributed["region"] = ...  # step 2: unknown label
known = attributed.dropna(subset=[...])  # step 3: drop, not fill
revenue = known[...].astype("float64").groupby([...]).sum()  # step 5: sum per region and month
return revenue.unstack(..., fill_value=0.0)  # step 6: wide, cells 0.0
```

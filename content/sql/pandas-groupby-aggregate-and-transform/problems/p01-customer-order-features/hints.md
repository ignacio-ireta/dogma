## Hint 1 — concept

Every feature describes one customer's orders in time order. Two steps come before any feature: put
the rows in time order, with `order_id` breaking ties on the same date, and then compute inside each
customer's group. Some features keep one value per order, and some reduce a whole group. For each
feature, ask which of the two shapes it needs.

## Hint 2 — formula / approach

A per-row value that uses the customer's total is a transform: it returns one value per row, aligned
to the row index. A per-row value that looks at the previous order of the same customer is a grouped
`shift`. A running sum inside a group is a grouped `cumsum`. Watch the dtypes: `cumcount` is int64 and
starts at 0, `rank` returns float64, and the day difference needs an explicit cast to float64, because
its dtype depends on whether the frame is empty.

## Hint 3 — code shape

```python
frame = orders.copy()
frame["order_date"] = pd.to_datetime(frame["order_date"])
frame = frame.sort_values(...).reset_index(drop=True)
by_customer = frame.groupby(...)
frame["order_seq"] = ...
frame["share_of_customer_total"] = ...
frame["days_since_previous_order"] = ...
frame["customer_running_total"] = ...
return frame[...]

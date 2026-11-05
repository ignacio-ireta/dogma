# Interview notes: pandas groupby, aggregate and transform

## Q1. For each order, how do you add the customer's order number, their share of spend and the days since their previous order?

**60-second answer.** Sort the orders by customer, date and order id first, because the sequence and
the gap only mean something in time order. Then group by customer. The order number is
`cumcount() + 1`. The share is `amount` divided by `transform("sum")`, which puts the customer's total
on each row. The gap is the date minus a grouped `shift()`, taken as whole days. Each result has one
value per order, so each is assigned straight back to the frame. I state the two NaN rules up front:
a first order has no gap, and a customer whose total is zero has no defined share.

**Follow-ups.**
- *Why sort before the groupby?* Inside a group, `cumcount`, `shift` and `cumsum` follow row order.
  Without the sort they follow arrival order.
- *What if an order has a NaN customer_id?* With the default `dropna=True`, that row gets NaN
  and is left out of every group. Decide whether it counts before you run the code.

## Q2. When would you use `apply`, and what does it cost?

**60-second answer.** `apply` is the general escape hatch. It calls a function once per group and
combines whatever it returns. When a built-in `transform` or `agg` fits, I use that to avoid a Python
callback per group. `transform` keeps the row index; `agg` returns group summaries. I reach for `apply` when the
per-group result is not one value per row, such as a small model per customer. The cost is a Python
call per group, and the result's index can gain the group keys, so the output shape needs checking.

**Follow-ups.**
- *What does `group_keys=False` change?* It stops the group keys being added to the result's index.
- *How would you test an `apply` function?* Run it on one group by hand, then on an empty frame, since
  the empty case can differ from the per-group case.

## Common mistakes
- Computing a running total or a gap before sorting by date.
- Breaking ties on the date alone, so orders on the same day come out in arrival order.
- Using `agg` where a per-row value is needed, then writing a summary back without a merge.
- Letting `rank` or `dt.days` set a dtype that changes with the data.
- Treating a zero total as a valid share of 0.0, rather than deciding NaN on purpose.

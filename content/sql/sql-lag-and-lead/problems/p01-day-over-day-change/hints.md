## Hint 1 — concept

Each account is its own ordered sequence of recorded days. Ask what "the row before this one"
means inside one account, and what the first row of each account should return.

## Hint 2 — formula / approach

`LAG(column)` returns the value from the previous row of the same partition, in the window's
order. Partition by the account, order by the date, and subtract the previous calls from the
current calls. The first row of a partition gets the function's default, which is NULL. Do not
add calendar days to the table.

## Hint 3 — code shape

Select the three input columns, then three window expressions over one named window: the previous
date, the previous calls, and their difference. End with `ORDER BY account_id, usage_date`, because
window order does not set the order of the returned rows.

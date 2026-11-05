## Hint 1 — concept

Each order is compared with a number that depends on its own customer. A subquery that refers to a
column of the outer row is a correlated subquery: it is evaluated for each outer row, and it sees
that row's customer.

## Hint 2 — formula / approach

Filter the candidates first: completed, amount not NULL, customer not NULL. For each candidate,
compute `AVG` of that customer's completed amounts in a correlated scalar subquery. Then keep the
rows whose amount is strictly greater than that average. A derived table lets you filter on the
computed column, and an `ORDER BY` on `order_id` makes the output repeatable.

## Hint 3 — code shape

```sql
SELECT ...
FROM (
    SELECT o.order_id, ..., (
        SELECT AVG(...) FROM orders AS i
        WHERE i.customer_id = o.customer_id AND ...   -- the correlation
    ) AS customer_average
    FROM orders AS o
    WHERE ...                                         -- the candidates
) AS scored
WHERE ...                                             -- strictly above the average
ORDER BY ...;
